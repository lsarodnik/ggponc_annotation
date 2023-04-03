def convert_to_spans(sentence, labels, tokenizer, id2label, special_token_mask, skip_subwords):
    sent_ids, sent_tags = sentence["input_ids"], labels
    
    output_spans = []
    
    sent_tokens = tokenizer.convert_ids_to_tokens(sent_ids)

    current_span = None
    start_index = 0
    is_subword = False

    for token, tag, special_token_mask in zip(sent_tokens, sent_tags, special_token_mask):    
        is_subword = token.startswith('##')
        if special_token_mask == 1:
            continue
        if start_index != 0 and not is_subword:
            start_index += 1
        clean_token = token[2:] if is_subword else token
        end_index = start_index + len(clean_token)
        label = id2label[tag] if tag != -100 else None
        if not label or (skip_subwords and is_subword):
            if current_span:
                current_span['word'] += (' ' if not is_subword else '') + clean_token
                current_span['end'] = end_index
        elif label == "O":
            if current_span:
                output_spans.append(current_span)
            current_span = None
        else: #Entity
            entity_type = label[2:]
            if label.startswith("B") and not is_subword:
                if current_span:
                    output_spans.append(current_span)
                    current_span = None
            elif label.startswith("I") or label.startswith("B"):
                if current_span and current_span['entity_group'] != entity_type:
                    output_spans.append(current_span)
                    current_span = None
            else:
                raise Exception(f"Only supporting IO and IOB tagging, unknown tag {label}")

            if not current_span:
                current_span = {
                    'entity_group' : entity_type,
                    'start' : start_index,
                    'end' : end_index,
                    'word' : clean_token,
                }
            else:
                assert current_span['entity_group'] == entity_type
                current_span['end'] = end_index
                current_span['word'] += (' ' if not is_subword else '') + clean_token

        start_index = end_index # Assuming white space tokenization

    if current_span:
        output_spans.append(current_span)

    return output_spans

def ner_error_analysis(sentence, gt_labels, pred_labels, special_tokens_mask, tokenizer, id2label, skip_subwords=False):
    gt_spans = convert_to_spans(sentence, gt_labels, tokenizer, id2label, special_tokens_mask, skip_subwords)
    pred_spans =  convert_to_spans(sentence, pred_labels, tokenizer, id2label, special_tokens_mask, skip_subwords)
    return _ner_error_analysis(pred_spans, gt_spans)

def _ner_error_analysis(pred_spans, gt_spans,gt_passages,analysis_type='complex'):
    def hf_to_tuples(spans):
        return [(s['start'], s['end'], s['entity_group'], s['word']) for s in spans]
    def passages_to_tuples(spans):
        return [(s['start'], s['end'], s['sentence'],s['sentence_id']) for s in spans]
    return ner_annotation_eval(hf_to_tuples(pred_spans), hf_to_tuples(gt_spans), passages_to_tuples(gt_passages),analysis_type)

from enum import Enum

class NERErrortype(Enum):
    TP = 'true_positive'
    FP = 'false_positive'
    FN = 'false_negative'
    LE = 'labeling_error'
    BE = 'boundary_error'
    BEs = 'boundary_error_smaller'
    BEl = 'boundary_error_larger'
    BEo = 'boundary_error_overlap'
    LBE = 'label_boundary_error'

def _get_ranges_labels(seq):
    return zip(*[((s[0], s[1]), s[2]) for s in seq])

def get_complex_error_types(pred_sq_list, gt_seq_list):

    error_types = []

    pred_sq_list_tmp = pred_sq_list.copy()
    gt_seq_list_tmp = gt_seq_list.copy()

    #remove entities from temp list when matched
    def rem_matched_entities(p_entity, gt_entity):
        if p_entity in pred_sq_list_tmp:
            pred_sq_list_tmp.remove(p_entity)
        if gt_entity in gt_seq_list_tmp:
            gt_seq_list_tmp.remove(gt_entity)

    #reduce entities by TP's (if any)
    for gt_seq in gt_seq_list:
        for pred_sq in pred_sq_list:
              if pred_sq == gt_seq:
                error_types.append([(pred_sq), (gt_seq), NERErrortype.TP])
                rem_matched_entities(pred_sq,gt_seq)
                break

    #reduce pred entities by BE's and LBE's (if any)
    for pred_sq in pred_sq_list_tmp[:]:     
        for gt_seq in gt_seq_list:

            #labels match -> BE, not matching -> LBE
            if gt_seq[2] == pred_sq[2]:
                #pred is smaller then GT
                if pred_sq[0] >= gt_seq[0] and pred_sq[1]<=gt_seq[1]:
                    error_types.append([(pred_sq), (gt_seq), NERErrortype.BEs])
                    rem_matched_entities(pred_sq,gt_seq)
                    break
                elif pred_sq[0] <= gt_seq[0] and pred_sq[1]>=gt_seq[1]:
                    error_types.append([(pred_sq), (gt_seq), NERErrortype.BEl])
                    rem_matched_entities(pred_sq, gt_seq)
                    break
                elif pred_sq[0] <= gt_seq[1] and gt_seq[0] <= pred_sq[1]:
                    error_types.append([(pred_sq), (gt_seq), NERErrortype.BEo])
                    rem_matched_entities(pred_sq, gt_seq)
                    break
            elif pred_sq[0] <= gt_seq[1] and gt_seq[0] <= pred_sq[1]:
                error_types.append([(pred_sq), (gt_seq),NERErrortype.LBE])
                rem_matched_entities(pred_sq, gt_seq)
                break
    
    #reduce GT entities by BE's and LBE's (if any)
    for gt_seq in gt_seq_list_tmp[:]:     
        for pred_sq in pred_sq_list:

            #labels match -> BE, not matching -> LBE
            if gt_seq[2] == pred_sq[2]:
                #pred is smaller then GT
                if pred_sq[0] >= gt_seq[0] and pred_sq[1]<=gt_seq[1]:
                    error_types.append([(pred_sq), (gt_seq), NERErrortype.BEs])
                    rem_matched_entities(pred_sq,gt_seq)
                    break
                elif pred_sq[0] <= gt_seq[0] and pred_sq[1]>=gt_seq[1]:
                    error_types.append([(pred_sq), (gt_seq),NERErrortype.BEl])
                    rem_matched_entities(pred_sq, gt_seq)
                    break
                elif pred_sq[0] <= gt_seq[1] and gt_seq[0] <= pred_sq[1]:
                    error_types.append([(pred_sq), (gt_seq),NERErrortype.BEo])
                    rem_matched_entities(pred_sq, gt_seq)
                    break
            elif pred_sq[0] <= gt_seq[1] and gt_seq[0] <= pred_sq[1]:
                error_types.append([(pred_sq), (gt_seq), NERErrortype.LBE])
                rem_matched_entities(pred_sq, gt_seq)
                break
    
    #remaining pred entities are FP's
    for p in pred_sq_list_tmp[:]:
        error_types.append([(p), (), NERErrortype.FP])
        rem_matched_entities(p, None)

    #remaining GT entities are FN's
    for gt in gt_seq_list_tmp[:]:
        error_types.append([(), (gt), NERErrortype.FN])
        rem_matched_entities(None, gt)

    assert len(pred_sq_list_tmp) == 0 and len(gt_seq_list_tmp)==0


    return error_types


def get_simple_error_types(pred_sq_list, gt_seq_list):
    error_types = []
    pred_sq_list_tmp = pred_sq_list.copy()
    gt_seq_list_tmp = gt_seq_list.copy()
    for gt_seq in gt_seq_list:
        #reduce detected GT entities by TP's (if any)
        for pred_sq in pred_sq_list:
              if pred_sq == gt_seq:
                error_types.append([(pred_sq), (gt_seq), NERErrortype.TP])
                pred_sq_list_tmp.remove(pred_sq)
                gt_seq_list_tmp.remove(gt_seq)

    for p in pred_sq_list_tmp:
        error_types.append([(p), (), NERErrortype.FP])
    
    for gt in gt_seq_list_tmp:
        error_types.append([(), (gt), NERErrortype.FN])

    return error_types





        
def ner_annotation_eval(predicted_entities, ground_truth_entities, ground_truth_sentences, analysis_type):

    results = []

    #sentence based processing
    for sentence in ground_truth_sentences:

        #reset matched entities per sentence
        current_sentence_gt_entities = []
        current_sentence_pred_entities = []

        #all gt's of the sentence
        for gt_entry in ground_truth_entities:
            if gt_entry[0] >= sentence[0] and gt_entry[1] <= sentence[1]:
                current_sentence_gt_entities.append(gt_entry)
         #all pred's of the sentence
        for pred_entry in predicted_entities:
            if pred_entry[0] >= sentence[0] and pred_entry[1] <= sentence[1]:
                current_sentence_pred_entities.append(pred_entry)
        
        #omit sentences without entities
        if not current_sentence_gt_entities and not current_sentence_pred_entities:
            continue

        #choose error_analysis type
        error_types = get_simple_error_types(current_sentence_pred_entities, current_sentence_gt_entities) if analysis_type=='simple' else get_complex_error_types(current_sentence_pred_entities, current_sentence_gt_entities)
        for error in error_types:
            results.append({
            'sentence_id': sentence[3],
            'predictions' : error[0],
            'matches': error[1],
            'categories': error[2].value
            })
    return results

def ner_check_error_type(predicted_entity, ground_truth_entities):
    # Option 1: false_positive  No match of the prediction to any
    # groud_truth annotations
    match = None
    category = 'false_positive'
    for ground_truth in ground_truth_entities:
        # Used to determine string overlap
        range_prediction = list(range(predicted_entity[0],
                                      predicted_entity[1] + 1))
        range_ground_truth = list(range(ground_truth[0],
                                        ground_truth[1] + 1))

        if predicted_entity == ground_truth:
            # Option 2: true_positive  Exact match of the prediction to
            # the groud_truth annotations
            match = ground_truth
            category = 'true_positive'
            break
        elif (predicted_entity[0] == ground_truth[0] and
              predicted_entity[1] == ground_truth[1]):
            # Option 3: labeling_error  Correct boundaries, but
            # incorrect label
            match = ground_truth
            category = 'labeling_error'
            break
        elif len([char_position for char_position in range_prediction
                  if char_position in range_ground_truth]) != 0:
            # There is an overlap
            # There could be an overlap with multiple entities. This
            # will be ignored as it is still a boundary error and does
            # not provide additoinal information
            if predicted_entity[2] == ground_truth[2]:
                # Option 4: boundary_error  Correct Label, but only
                # overlapping boundaries
                match = ground_truth
                category = 'boundary_error'
                break
            else:
                # Option 5: labeling_boundary_error  Incorrect label,
                # but overlapping boundaries
                match = ground_truth
                category = 'label_boundary_error'
                break
    return category, match