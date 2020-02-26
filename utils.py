import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
import os, sys
import pandas as pd
import numpy as np
import shap
import sqlalchemy as db
from TestingExceptions import * 

import locale
locale.setlocale( locale.LC_ALL, '' )

LABELS = ["No", "Yes"]
col_list = ["cerulean", "scarlet"]
sns.set(style='white', font_scale=1.75, palette=sns.xkcd_palette(col_list), color_codes=False)

def axis2display(axis_name):
    return ' '.join(map(lambda x: x.capitalize(), axis_name.split('_')))
                    
def plot_2d(X, y, x_axis, y_axis, npoints = 300):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(axis2display(x_axis), fontsize = 15)
    ax.set_ylabel(axis2display(y_axis), fontsize = 15)
    ax.set_title('Visualise dataset (%d points)'%npoints, fontsize = 20)
    targets = ['no', 'yes']
    colors = ['r', 'g']
    for num, target, color in zip([0,1], targets,colors):
        indicesToKeep = y == num
        ax.scatter(X.head(npoints).loc[indicesToKeep, x_axis]
                   , X.head(npoints).loc[indicesToKeep, y_axis]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    
def plot_roc(test_y, pred_proba):
    false_pos_rate, true_pos_rate, thresholds = roc_curve(test_y, pred_proba)
    roc_auc = auc(false_pos_rate, true_pos_rate,)

    plt.figure(figsize=(5, 5))
    plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
    plt.plot([0,1],[0,1], linewidth=5)

    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='lower right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    
def drop_id(X_input):
    """Drop the participant_id column."""
    if 'participant_id' in X_input.columns.values:
        X_input.drop('participant_id', axis=1, inplace=True)
    return X_input


def get_test_ids(X_train, X_test, X_pred):
    """Get the participant ids in the test set to produce the final output."""
    test_pid = X_test.participant_id
    X_train, X_test, X_pred = drop_id(X_train), drop_id(X_test), drop_id(X_pred)
    return X_train, X_test, X_pred, test_pid    


def train_val_test_split(X, y, test_size=0.1, validation_size=0.2, random_state=42):
    """Split the data into train/validation/test sets, 
    in the ratio of 70%/20%/10%, remember the second split 
    requires different ratios."""
    size = test_size + validation_size
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=test_size / size, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_labelled(df, X):
    """Add participant id column and remove special characters."""
    X['participant_id'] = df.participant_id.astype(int)
    not_null = df.created_account.notnull()
    to_replace = ['?', '-', ' ', '(', ')', '&']
    X.columns = X.columns.str.replace('?', 'unknown')
    for symb in to_replace:
        X.columns = X.columns.str.replace(symb, '_')
    # Split the data into modelling set and prediction set
    X_model = X[not_null]
    y_model = df[not_null].created_account.astype(int)
    X_pred = X[~not_null]
    return X_model, y_model, X_pred


def evaluate(test_x,test_y,model,preds=None, probas=None, top_n = None,
             print_cm=True, print_roc=False, print_prec_rec = False, 
             prec_k=100, perform_cost_benefit =False):
    if preds is None:
        preds = model.predict(test_x)
    conf_matrix = confusion_matrix(test_y, preds)
    #print(conf_matrix)
    
    if probas is None:
        probas = model.predict_proba(test_x)[:,1]
     
        
    if top_n != None:
        aa = list(zip(probas,test_y))
        aa.sort(key=lambda x: x[0], reverse=True)
        aa = aa[:int(len(aa)*top_n)]
        result = ([ a for a,b in aa ], [ b for a,b in aa ])
        ap = average_precision_score(result[1], result[0])
        preds = [int(round(i)) for i in result[0]]
        conf_matrix = confusion_matrix(result[1], preds)
        print("Average Precision for the top %.2d percent (AURPC) is %.2f" %(top_n*100, ap))
        
    else:
        ap = average_precision_score(test_y, probas)
        print('Average Precision (AURPC) is %.2f' % ap)
    
    if print_cm:
        plot_conf_matrix(conf_matrix)
        
    if print_roc:
        plot_roc(test_y,probas)
    
    if print_prec_rec:
        precision_rt, recall_rt, threshold_rt = precision_recall_curve(test_y, probas)
        plt.figure(figsize=(5,5))
        plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
        plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
        plt.title('Precision and recall for different threshold values')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        plt.show()
    return

def precision_at_k(y_true, y_pred, k=10):
    a = list(zip(y_pred,y_true))
    a.sort(reverse=True)
    return sum(map(lambda x: x[1], a[:k])) / k

def columns_forBaseline(df):
    text_columns = df.columns[df.columns.str.contains('text_')]#[1:]
#    X_baseline = df.drop(["escalation",'Consumer.complaint.narrative','lower','modified_lower','Unnamed: 0','Unnamed: 0.1'], axis = 1)
    X_baseline = df.drop(text_columns, axis=1)
    return X_baseline

def gen_model_data(df, use_data_type='survey_salary_occupation'):
    if use_data_type == 'nw_results_only':
        X = df[['familiarity_nw', 'view_nw']]
    elif use_data_type == 'survey_only':
        X = pd.get_dummies(df[['age', 'marital_status', 'education_num', 'familiarity_nw', 'view_nw']])
    elif use_data_type == 'survey_salary':
        X = pd.get_dummies(df[['age', 'marital_status', 'education_num', 'familiarity_nw', 'view_nw', 
                           'years_with_employer', 'hours_per_week', 'capital_gain', 'capital_loss']])
    elif use_data_type == 'survey_salary_occupation':
        X = pd.get_dummies(df[['age', 'marital_status', 'education_num', 'familiarity_nw', 'view_nw', 
                           'years_with_employer', 'hours_per_week', 'capital_gain', 'capital_loss', 
                           'occupation_level']])
    elif use_data_type == 'use_additional':
        df = pd.read_csv('data/extended_data_for_modelling.csv')
        X = pd.get_dummies(df[['age', 'marital_status', 'education_num', 'familiarity_nw', 'view_nw', 
                           'years_with_employer', 'hours_per_week', 'capital_gain', 'capital_loss', 
                           'occupation_level', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'dcv']])
    else:
        raise NotImplemented
    y = df.created_account == 'Yes'
    return X, y


def plot_confusion_matrix(y_test, preds):
    conf_matrix = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    
    
# Plot confusion matrix with specified labels
def gen_labels(conf_matrix):
    total = np.sum(conf_matrix)
    labels = []
    for values in conf_matrix:
        inner_lab = []
        for val in values:
            str_lab = ' {0:d}\n\n({1:.1f}%)'.format(val, val/total*100)
            inner_lab.append(str_lab)
        labels.append(inner_lab)
    labels = np.array(labels)
    return labels

    
def plot_conf_matrix(conf_matrix):
    """Plot confusion matrix with Mudano colors."""
    import seaborn as sns
    from matplotlib import ticker
    from matplotlib.colors import LinearSegmentedColormap

    # Generate confusion matrix
    labels = gen_labels(conf_matrix)
    boundaries = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # custom boundaries
    data = np.array([[0.5, 0], [0, 0.5]])
    
    hex_colors = sns.light_palette('navy', n_colors=len(boundaries) * 2 + 2, as_cmap=False).as_hex()
    hex_colors = [hex_colors[i] for i in range(0, len(hex_colors), 2)]

    colors=list(zip(boundaries, hex_colors))

    custom_color_map = LinearSegmentedColormap.from_list(
        name='custom_navy',
        colors=colors,
    )
    plt.figure(figsize=(5, 5))
    sns.heatmap(vmin=0.0,vmax=1.0,
        data=data,
        fmt='',
        cmap=custom_color_map,
        xticklabels=LABELS,
        yticklabels=LABELS,
        linewidths=0.75,
        annot=labels,
        annot_kws={"size": 16},
        cbar=False,
    )
    #plt.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
    #plt.gca().yaxis.set_major_formatter(ticker.IndexLocator(base=1, offset=0.5))
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    

def save_output(model, X_test, test_pid, filename, y_test, old_participant_ids=None):
    """
    Use model to make predictions, calculate prediction probabilities,
    calculate feature importance and save to file.
    """
    import shap
    X_out = X_test.copy()
    y_pred = model.predict(X_test)
    X_out['predicted'] = y_pred
    y_prob0 = model.predict_proba(X_test)[:,0]
    y_prob1 = model.predict_proba(X_test)[:,1]
    X_out['proba0'] = y_prob0
    X_out['proba1'] = y_prob1
    X_out['participant_id'] = test_pid.values
    X_out['created_account'] = y_test.values
    if old_participant_ids is not None:
        X_out['old_participant_id'] = old_participant_ids.values
    X_out['actual'] = y_test.values
    
    explainer = shap.TreeExplainer(model) # insert algorithm to explain here
    shap_values = explainer.shap_values(X_test)
    
    df = pd.DataFrame(shap_values[1])
    df.columns = X_test.columns
    df['participant_id'] = test_pid.values
    df.participant_id = df.participant_id.astype(int)
    #df['actual'] = y_test.values
    
    df = df.merge(X_out, on='participant_id')
    df.set_index('participant_id', inplace=True)
    df.to_csv(filename)
    print(df.columns)
    return df

    
def save_feature_importance_rf(rf, X_test, fname):
    """Use the .feature_importances_ attribute in the Random Forest
    to evaluate the importance of each features used to build
    the model. Visualize as horizontal bar chart.
    """
    f_imp = pd.DataFrame()
    f_imp['features'] = X_test.columns
    f_imp['importances'] = rf.feature_importances_
    f_imp.set_index('features', inplace=True)
    f_imp.sort_values(by='importances', ascending=False, inplace=True)
    f_imp.to_csv(fname)
    return f_imp


def gen_feature_names(data_file=None, df=None):
    """Generate all the features from DataFrame"""
    if data_file:
        df = pd.read_csv(data_file)
    
    features = print("features = [")
    num_cols = ['age', 
            'education_num', 
            'avg_curr_account_balance', 
            'avg_overdraft_balance', 
            'hours_per_week', 
            'occupation_level',
            'familiarity_dm', 
            'view_dm', 
            'months_with_employer',
            'years_with_employer', 
            'loan_initial_amount', 
            'loan_outstanding_balance',
             'mort_age']
    cat_cols = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "race",
            "sex",
            "native_country",
            "name_title",
            "first_name",
            "last_name",
            "full_name",
            "postcode",
            "town",
            "job_title",
            "company_email",
            "dob",
            "interested_insurance",
            "paye",
            "salary_band",
            "religion",
            "new_mortgage",
            ]
    for c in df.columns:
        print('    "{0: <30} # '.format(str(c)+'",') + ('Categorical' if c in cat_cols else 'Numerical' if c in num_cols else ''))
    #for c in cat_cols:
    #    print('    "{0: <30} # Categorical'.format(str(c)+'",'))
    print(']')
    return df.columns


def reduce_categorical_options(df, max_options=10, feature_of_interest='town'):
    """
    Allow for a maximum number of most frequent categorical values and replace all other with value "Other"
    This assumes the type of categorical values is string
    """
    if feature_of_interest not in df.columns:
        print('feature not in dataframe')
        return df
    allowed = df[feature_of_interest].value_counts().head(max_options).index.tolist()
    df[feature_of_interest] = df[feature_of_interest].apply(lambda x: x if x in allowed else 'Other')
    return df
    

def plot_cost_benefit(data, Current_spent, show_best=False):
    """
    Plot cost benefit analysis against proportion of customers contacted (pred=1)
    """
    utility = [0] + list(data.saving)#/ 100
    
    NUM_SAMPLES=len(data)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid('on')
    x_axis = np.linspace(0, 100, NUM_SAMPLES + 1)
    ax.fill_between(x_axis, utility, facecolor='navy', alpha=.7, 
                    label='Expected utility')
    ax.plot(x_axis, utility, color='navy', alpha=.7)

    ax.set_xlim([0, 100])
    ax.set_xticks(np.arange(0, 100, 10.0))
    
    #ax.set_ylim([(Current_spent + (Current_spent*0.25)), 0])
    #ax.spines['left'].set_position(('data', 0))
    #ax.spines['bottom'].set_position(('data', Current_spent))
    #ax.xaxis.labelpad = 80
    #ax.set_ylabel('Cost  [ £k ]')  
    
    ax.set_xlabel('% of Automated Queries')
       
    ax.set_ylabel('% of cost saved')
    
    ax.legend()
    ax.grid(True)
    plt.title('Utility Cost Benefit Analysis')
    plt.tight_layout()
    plt.show()
    plt.savefig('figures/cost_benefit_analysis.png', bbox_inches='tight')
    if show_best:
        lis = [(x_axis[i], utility[i]) for i in range(len(x_axis))]
        lis = lis[1:]
        best = max(lis, key=lambda x: x[1])
        
        utility_number = [0] + list(data.cumul_cost)
        lis2 = [(x_axis[i], utility_number[i]) for i in range(len(x_axis))]
        lis2 = lis2[1:]
        #print(lis2)
        #print(type(lis2))
        best2 = max(lis2, key=lambda x: x[1])
        max_save_rate = round(best[1],2)
        max_aut_queries = best[0]
        amount_saved = best[1]*(-Current_spent)/100
        print(f'Maximum saving rate is {round(best[1],2)}% if {best[0]:.2f}% of queries is automated')
        print(f'This accounts for {locale.currency(best[1]*(-Current_spent)/100)} saved')
        #print(f'This accounts for {locale.currency(best2[1])} saved')
        return max_save_rate, max_aut_queries, amount_saved

def cost_benefit_analysis(y_true, y_pred,
        perc_contacted, 
        total_applicable_customers,
        cost_of_reviewing_query,
        cost_of_false_negative, 
        do_eval=False,
        plot_utility=False,
        show_best=False):
    """Run the evaluate function only on the contacted subset.
    Clearly there will be no TN or FN values. Is this correct?
    
    Returns
    -------
    reward - of the cost model at a threshold for these probas
    data - dataframe with cumulative cost and values sorted by probas
    """
    
    if type(cost_of_reviewing_query) not in (int, float):
        raise TypeError(f'cost_of_reviewing_query {cost_of_reviewing_query} not valid. Please input a number.')  
    elif cost_of_reviewing_query <0:
        raise InputError(f'cost_of_reviewing_query{cost_of_reviewing_query} less than 0. Please input 0 or a positive number.') 
  
    if type(cost_of_false_negative) not in (int, float):
        raise TypeError(f'cost_of_false_negative {cost_of_false_negative} not valid. Please input a number.')
    elif cost_of_false_negative <0:
        raise InputError(f'cost_of_false_negative{cost_of_false_negative} less than 0. Please input 0 or a positive number.') 
 
    if (perc_contacted <0 or perc_contacted>100):
        raise PercentageError(f"Percentage value {perc_contacted} outside of the limits. Please input a number between 0 and 100.") 
        
    if total_applicable_customers <=0:
        raise InputError(f"Number of total customers cannot be less than 0 or 0. Current number: {total_applicable_customers}")
       
    
    #print('exceptions run')
    VALUE_TRUE_POSITIVE = cost_of_reviewing_query
    VALUE_FALSE_POSITIVE = (-cost_of_reviewing_query)
    VALUE_TRUE_NEGATIVE = (-cost_of_reviewing_query)
    VALUE_FALSE_NEGATIVE = (-cost_of_false_negative)
    
    Current_spent = VALUE_FALSE_POSITIVE*total_applicable_customers
    Current_spent *= total_applicable_customers / len(y_true)
    print(f'Current spending in pounds: {locale.currency(abs(Current_spent))}')
    #print('Current_spent')
    
    k = int(len(y_true)*perc_contacted / 100)
    a = list(zip(y_pred,y_true))
    a.sort(key=lambda x: x[0], reverse=True)
    
    true_positives = sum(map(lambda x: x[1], a[:k]))
    false_positives = k - true_positives
    false_negatives = sum(map(lambda x: x[1], a[k:]))
    true_negatives = len(y_true) - k - false_negatives
        
    k = int(perc_contacted*len(y_pred) /100)
    a = list(zip(y_pred,y_true.values))
    a.sort(key=lambda x: x[0], reverse=True)
    data = pd.DataFrame(a, columns=['pred_proba', 'true'])
    
    TPFPTNFN = pd.DataFrame(columns=['TP','FP','TN','FN'])
    
    for i in range(len(a)):
        if (round(a[i][0])==1 and a[i][1]==1):
            TPFPTNFN = TPFPTNFN.append({'TP': 1}, ignore_index=True)
        elif (round(a[i][0])==0 and a[i][1]==0):
                TPFPTNFN = TPFPTNFN.append({'TN': 1}, ignore_index=True)
        elif (round(a[i][0])==1 and a[i][1]==0):
                TPFPTNFN = TPFPTNFN.append({'FP': 1}, ignore_index=True)
        elif (round(a[i][0])==0 and a[i][1]==1):
                TPFPTNFN = TPFPTNFN.append({'FN': 1}, ignore_index=True)
                
    TPFPTNFN = TPFPTNFN.fillna(0, inplace=False)
    
    data['cost'] = TPFPTNFN.apply(lambda x: VALUE_TRUE_POSITIVE if x['TP'].astype(int) \
                                     else (VALUE_FALSE_POSITIVE if x['FP'].astype(int) \
                                      else (VALUE_TRUE_NEGATIVE if x['TN'].astype(int) \
                                      else VALUE_FALSE_NEGATIVE)), axis=1)
    
    data['cost'] *= total_applicable_customers / len(y_true)
    data['cumul_cost'] = data.cost.cumsum()
    
   # adj = list()
   # for i in range(len(y_true)):
   #     adj.append(data['cumul_cost'][i] - abs(Current_spent * (1-i/len(y_true))))
   # data['adj'] = adj 
    
    #data['saving'] = (1-data['adj']/Current_spent) * 100
    data['saving'] = data['cumul_cost']/abs(Current_spent) * 100
    
    if perc_contacted == 0:
        print("Gain by automating %d%% of the queries in Pounds: %s" % (perc_contacted, locale.currency(Current_spent)))
    else:
        perc_idx = int(perc_contacted*len(data)/100)
        if perc_idx == 0:
            perc_idx = 1
        #print(f'perc_idx={perc_idx}')
        print("Savings by automating %d%% of the queries in Pounds: %s" 
              % (perc_contacted,locale.currency(data['cumul_cost'][perc_idx-1])))  
         
    if do_eval:
        evaluate( None, data['true'], None, 
             preds=[1] * k + [0] * (len(data) - k), 
             probas=list(data['pred_proba'].values))
    
    if plot_utility:
        plot_cost_benefit(data, Current_spent, show_best)
    
    return data, TPFPTNFN, Current_spent

# Shaply plots
def visualize_feature_importance_shap(shap_values, X_test):
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
def visualize_feature_importance_violin(shap_values, X_test):
    shap.summary_plot(shap_values, X_test, alpha = .6)



from sqlalchemy import event, create_engine
from sqlalchemy import create_engine, MetaData
from io import StringIO


def get_engine(con_string):
    engine = create_engine(con_string
                          )#, echo=True)
    #@event.listens_for(engine, 'before_cursor_execute')
    #def plugin_bef_cursor_execute(conn, cursor, statement, params, context,executemany):
    #    if executemany:
    #        cursor.fast_executemany = True  # replace from execute many to fast_executemany.
    #        cursor.commit()
    return engine


def cleanColumns(columns):
    cols = []
    for col in columns:
        col = col.replace(' ', '_').replace("'",'_').replace('.','_').replace('/','_')
        cols.append(col)
    return cols

def cleanRows(df, text_cols, double_cols):
    for col in text_cols:
        df[col] = df[col].str.replace(',', ' ')
    for col in double_cols:
        df.loc[df[col].isna(), col] = 0#'NULL'
    return df

def read_joined(table_name, con):
    engine = db.create_engine(con)
    df = pd.read_sql_query('select * from {}'.format(table_name), con=engine)
    return df


def to_postgres(df, table_name, con):
    data = StringIO()
    df.columns = cleanColumns(df.columns)
    text_cols = []
    double_cols = []
    if 'vendor_name' in df.columns:
        text_cols.append('vendor_name')
    if 'job_title' in df.columns:
        text_cols.append('job_title')
    if 'longitude' in df.columns:
        double_cols.append('longitude')
    if 'latitude' in df.columns:
        double_cols.append('latitude')
    if 'replace_value' in df.columns:
        double_cols.append('replace_value')
    df = cleanRows(df, text_cols=text_cols, double_cols = double_cols)
    if df.index.name == 'participant_id':
        df.reset_index(inplace=True)
    df.to_csv(data, header=False, index=False)
    data.seek(0)
    raw = con.raw_connection()
    curs = raw.cursor()
    try:
        curs.execute("DROP TABLE " + table_name)
    except:
        raw = con.raw_connection()
        curs = raw.cursor()
        print("`%s` doesn't exist - CREATING" %table_name)
    empty_table = pd.io.sql.get_schema(df, table_name, con = con)
    empty_table = empty_table.replace('"', '')
    #print(empty_table)
    curs.execute(empty_table)
    curs.copy_from(data, table_name, sep = ',')
    curs.execute("grant select on %s to grp_dev"% table_name)
    curs.execute("grant select on %s to fil_dev"% table_name)
    curs.connection.commit()
    
def svd_inverse(argument_no, num_of_words):

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    import pickle
    
    #complaints = list(All_with_narrative['modified_lower'].values)
    #
    ##corpus = df[['Consumer.complaint.narrative']].values[0]
    #vectorizer = CountVectorizer(analyzer = 'word',
    #                            stop_words = 'english',
    #                            max_features = 1000,
    #                            ngram_range=(2, 2))
    #
    #Y = vectorizer.fit_transform(complaints)
    ##print(vectorizer.get_feature_names())
    ##Y = np.sort(Y.toarray())
    #
    #feature_array = np.array(vectorizer.get_feature_names())
    #tfidf_sorting = np.argsort(Y.toarray()).flatten()[::-1]
    #feature_array[tfidf_sorting]
    #
    #from sklearn.decomposition import TruncatedSVD
    #from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    #
    #tfidf = TfidfTransformer()
    #
    #svd = TruncatedSVD(n_components = 100,
    #            n_iter = 7,
    #            random_state = 0)
    #
    #z = tfidf.fit_transform(Y)
    #
    #trans = svd.fit_transform(z)
    trans = np.load('svd_transformed.npy')
    # To load again
    svd = pickle.load(open('svd_model.p', 'rb'))
    vectorizer = pickle.load(open('vectorizer_model.p', 'rb'))
    feature_array = np.array(vectorizer.get_feature_names())
        
    z_inverse = svd.inverse_transform(trans)
    
    return [feature_array[i] for i in svd.components_[argument_no].argsort()[::-1]][0:num_of_words]


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step
        
        
def evaluate_prc(test_x,test_y,model,preds=None, probas=None, quantiles_n = None,
         print_cm=True, print_roc=False, print_prec_rec = False, 
         prec_k=100, perform_cost_benefit =False):
    if preds is None:
        preds = model.predict(test_x)
    conf_matrix = confusion_matrix(test_y, preds)
    if probas is None:
        probas = model.predict_proba(test_x)[:,1]
        
    if quantiles_n != None:
        condition = probas>=list(np.quantile(model.predict_proba(test_x), [quantiles_n]))[0]
        ap = average_precision_score(test_y[condition], probas[condition])
        #conf_matrix = confusion_matrix(test_y[condition], preds[condition])
        #print(ap)
    else:
        ap = average_precision_score(test_y, probas)
    return ap

        
def precision_quant(baseM, textM, X_test_B, y_test_B, X_test_T, y_test_T, quantiles_n = None):
    
    avg_prec_b = list()
    avg_prec_t = list()
    quantiles = list()
    
    if quantiles_n == None:
        quantiles_n = 0.001
        
    for i in frange(quantiles_n, 1.0, 0.05):
        baseline_model_p = evaluate_prc(X_test_B, y_test_B, baseM, quantiles_n=i)
        avg_prec_b.append(baseline_model_p)
        
        Textmodel_model_p = evaluate_prc(X_test_T, y_test_T, textM, quantiles_n=i)
        avg_prec_t.append(Textmodel_model_p)
        
        quantiles.append(i)
        
    baseline_model_p = evaluate_prc(X_test_B, y_test_B, baseM, quantiles_n=0.99)
    #print(baseline_model_p)
    avg_prec_b.append(baseline_model_p)
    
    Textmodel_model_p = evaluate_prc(X_test_T, y_test_T, textM, quantiles_n=0.99)
    avg_prec_t.append(Textmodel_model_p)
    quantiles.append(0.99)
        
    plt.figure(figsize=(5,5))
    plt.plot(quantiles, avg_prec_b, label="Baseline",linewidth=3)
    plt.plot(quantiles, avg_prec_t, label="TextModel",linewidth=3)
    plt.title('Precision comparison')
    plt.xlabel('Quantiles')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.xlim((quantiles[1],1))
    plt.xticks(rotation=45)
    plt.show()
    return


