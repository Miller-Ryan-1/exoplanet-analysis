def matrix_result_2d(df, TP='NW', actual='rows', output='dict'):
    '''
    This function converts a 2x2 DataFrame (either from Crosstab or a Confusion Matrix) and outputs it's evaluation metrics.
    Input: It defaults to take in a dataframe with the *actual* values in the rows and predicted in columns, this can be changed by altering the "actual" parameter.
    Output: It defaults to output a dict object (for use in comparison dataframe), but can also be set to output a series of printed statements.
        Lastly, it defaults to the the True Positive being in the upper left quadrant (northwest/NW); parameter can be changed to lower right (SE).
    '''   
    if TP=='NW':
        if actual=='rows':
            true_positive = df.iloc[0,0]
            false_positive = df.iloc[1,0]
            true_negative = df.iloc[1,1]
            false_negative = df.iloc[0,1]
        elif actual=='columns':
            true_positive = df.iloc[0,0]
            false_positive = df.iloc[0,1]
            true_negative = df.iloc[1,1]
            false_negative = df.iloc[1,0]
        else:
            print('''If actual results (from training data) is in columns, please include "actual = 'columns'" in the function parameters''')
            return
    elif TP=='SE':
        if actual=='rows':
            true_positive = df.iloc[1,1]
            false_positive = df.iloc[0,1]
            true_negative = df.iloc[0,0]
            false_negative = df.iloc[1,0]
        elif actual=='columns':
            true_positive = df.iloc[1,1]
            false_positive = df.iloc[1,0]
            true_negative = df.iloc[0,0]
            false_negative = df.iloc[0,1]
        else:
            print('''If actual results (from training data) is in columns, please include " actual = 'columns' " in the function parameters''')
            return
    else:
        print('''If True Positive is in the bottom right quadrant, please include " TP = 'SE' " in the function parameters''')
        return
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    f1_score = (2*precision*recall)/(precision + recall)
    support_pos = true_positive + false_negative
    support_neg = true_negative + false_positive

    if output == 'dict':
        return {'true_positive':true_positive, 'false_positive':false_positive, 'true_negative':true_negative, 'false_negative':false_negative, 'precision':precision, 'recall':recall, 'accuracy':accuracy, 'f1_score':f1_score, 'support_pos':support_pos, 'support_neg':support_neg}
    elif output == 'print':
        print(f'True Positive = {true_positive}')
        print(f'False Positive = {false_positive}')
        print(f'True Negative = {true_negative}')
        print(f'False Negative = {false_negative}')
        print(f'Precision = {precision:.2f}')
        print(f'Recall = {recall:.2f}')
        print(f'Accuracy = {accuracy:.2f}')
        print(f'F1 Score = {f1_score:.2f}')
        print(f'Support, Positive = {support_pos}')
        print(f'Support, Negative = {support_neg}')
    else:
        return
    
