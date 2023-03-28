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

def ner_error_analyis(sentence, gt_labels, pred_labels, special_tokens_mask, tokenizer, id2label, skip_subwords=False):
    gt_spans = convert_to_spans(sentence, gt_labels, tokenizer, id2label, special_tokens_mask, skip_subwords)
    pred_spans =  convert_to_spans(sentence, pred_labels, tokenizer, id2label, special_tokens_mask, skip_subwords)
    return _ner_error_analyis(pred_spans, gt_spans)

def _ner_error_analyis(pred_spans, gt_spans):
    def hf_to_tuples(spans):
        return [(s['start'], s['end'], s['entity_group'], s['word']) for s in spans]
    return ner_annotation_eval(hf_to_tuples(pred_spans), hf_to_tuples(gt_spans))

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
    BEsc = 'boundary_error_span_count'
    LBE = 'label_boundary_error'

def _get_ranges_labels(seq):
    print(seq)
    return zip(*[((s[0], s[1]), s[2]) for s in seq])

def get_complex_error_types(pred_sq_list, gt_seq_list):
    error_types = []

   #reduce detected GT entities by TP's (if any)
    for gt_seq in gt_seq_list:
        
        for pred_sq in pred_sq_list:
              if pred_sq == gt_seq:
                error_types.append(NERErrortype.TP)
                pred_sq_list.remove(pred_sq)
                gt_seq_list.remove(gt_seq)
                break
    
    #reduce detected GT entities by LE's (if any)
    for gt_seq in gt_seq_list:          
        for pred_sq in pred_sq_list:
            pred_ranges, pred_labels = _get_ranges_labels(pred_sq)
            gt_ranges, gt_labels = _get_ranges_labels(gt_seq)
            if pred_ranges == gt_ranges:
                assert pred_labels != gt_labels
                error_types.append(NERErrortype.LE)
                pred_sq_list.remove(pred_sq)
                gt_seq_list.remove(gt_seq)
                break
    #reduce detected GT entities by Be's and LBE's (if any)
    for gt_seq in gt_seq_list:     
        for pred_sq in pred_sq_list:
            pred_ranges, pred_labels = _get_ranges_labels(pred_sq)
            gt_ranges, gt_labels = _get_ranges_labels(gt_seq)
            if all([l1 == l2 for l1 in pred_labels for l2 in gt_labels]):
                if len(pred_ranges) != len(gt_ranges):
                    error_types.append(NERErrortype.BEsc)
                    pred_sq_list.remove(pred_sq)
                    gt_seq_list.remove(gt_seq)
                    break
                elif all([p1 >= g1 and p2<=g2 for (p1, p2) in pred_ranges for (g1, g2) in gt_ranges]):
                    error_types.append(NERErrortype.BEs)
                    pred_sq_list.remove(pred_sq)
                    gt_seq_list.remove(gt_seq)
                    break
                elif all([p1 <= g1 and p2>=g2 for (p1, p2) in pred_ranges for (g1, g2) in gt_ranges]):
                    error_types.append(NERErrortype.BEl)
                    pred_sq_list.remove(pred_sq)
                    gt_seq_list.remove(gt_seq)
                    break
                elif all([p1 <= g2 and g1 <= p2 for (p1, p2) in pred_ranges for (g1, g2) in gt_ranges]):
                    error_types.append(NERErrortype.BEo)
                    pred_sq_list.remove(pred_sq)
                    gt_seq_list.remove(gt_seq)
                    break
            elif all([p1 <= g2 and g1 <= p2 for (p1, p2) in pred_ranges for (g1, g2) in gt_ranges]):
                error_types.append(NERErrortype.LBE)
                pred_sq_list.remove(pred_sq)
                gt_seq_list.remove(gt_seq)
                break

    #add FN error, for all left GT entities
    for gt_seq in gt_seq_list:
        error_types.append(NERErrortype.FN)
        gt_seq_list.remove(gt_seq)

    #add FP error, for all left Pred entities
    for pred_sq in pred_sq_list:
        error_types.append(NERErrortype.FP)
        pred_sq_list.remove(pred_sq)
        
    assert len(pred_sq_list) == 0 and len(gt_seq_list)==0

    return error_types
 




def get_simple_error_types(pred_sq_list, gt_seq_list):
    error_types = []
   
    for gt_seq in gt_seq_list:
        #reduce detected GT entities by TP's (if any)
        for pred_sq in pred_sq_list:
              if pred_sq == gt_seq:
                error_types.append(NERErrortype.TP)
                pred_sq_list.remove(pred_sq)
                gt_seq_list.remove(gt_seq)

    for _ in pred_sq_list:
        error_types.append(NERErrortype.FP)
    
    for _ in gt_seq_list:
        error_types.append(NERErrortype.FN)

    return error_types


        
def ner_annotation_eval(predicted_entities, ground_truth_entities, analysis_type='complex'):

    results = []
    subsequence = ([],[])

    start_index = 0
    end_index = 0
    matched_predicted_entities = []
    matched_gt_entities = []


    for current_gt_i,current_gt in enumerate(ground_truth_entities):

        #initialize first gt_entity
        if start_index == 0: start_index = current_gt[0] 
        if end_index == 0: end_index = current_gt[1]

        #check if current_gt.start <= end_index && current_gt.end >= start_index
        if current_gt[0] <= end_index and current_gt[1] >= start_index:
            if current_gt[0] < start_index: start_index = current_gt[0]
            if current_gt[1] > end_index: end_index = current_gt[1]
            subsequence[1].append(current_gt)
            matched_gt_entities.append(current_gt)

        #leave loop, calc all preds in span, append result and reset for next span
        if current_gt[0] > end_index or current_gt_i == len(ground_truth_entities)-1:
            #iterate over all pred entities for the current span (start / end)
            for current_pred_i, current_pred in enumerate(predicted_entities):
                #check if current_pred.start <= end_index && current_pred.end >= start_index
                if current_pred[0] <= end_index and current_pred[1] >= start_index:
                    #if current_pred[0] < start_index: start_index = current_pred[0]
                    #if current_pred[1] > end_index: end_index = current_pred[1]
                    subsequence[0].append(current_pred)
                    matched_predicted_entities.append(current_pred) #FP are later diffed based on this
            error_types = get_simple_error_types(subsequence[0], subsequence[1]) if analysis_type=='simple' else get_complex_error_types(subsequence[0], subsequence[1])
            results.append({
            'prediction' : subsequence[0],
            'match' : subsequence[1],
            'category' : error_types
            })

            #reset temp vars for iteration
            subsequence = ([],[])
            subsequence[1].append(current_gt)
            start_index = current_gt[0]
            end_index = current_gt[1]
        
    #handle FP cases
    unmatched_predicted_entities = list(set(predicted_entities) - set(matched_predicted_entities))
    for unmatched_pred in unmatched_predicted_entities:
            error_types = get_simple_error_types(unmatched_pred, None) if analysis_type=='simple' else get_complex_error_types(unmatched_pred, None)
            results.append({
            'prediction' : [unmatched_pred],
            'match' : [],
            'category' : error_types
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