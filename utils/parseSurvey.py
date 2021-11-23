import pandas as pd
import os
import yaml
import numpy as np
from copy import deepcopy
import pdb

import argparse
import subprocess
import inspect

from termcolor import colored
import itertools
from guppy import hpy
import warnings

import gc
import matplotlib.pyplot as plt


class SurveyParser(object):
    
    default_args = {
        'separator': ',',
        'ignore_files_with_phrases': ['codebook', 'codes_labels']
    }
    
    def get_xls_df_list(self, 
                        _data_files, 
                        **kwargs):
        survey_df_dict = {}
        remove_files_with_phrases = kwargs.default_args['ignore_files_with_phrases'] if 'ignore_files_with_phrases' in kwargs.keys() else self.default_args['ignore_files_with_phrases']
        data_files = [_file for _file in _data_files if not any(phrase in _file for phrase in remove_files_with_phrases)]

        for _file in data_files:
            try:
                sheet = pd.read_excel(_file, sheet_name=0) # Access the first sheet the other are codes and questionaire interpretations
            except Exception as e:
                print('Exception caught in xls')

                _csv = '.'.join(_file.split('.')[:-1])+'.csv'
                if os.path.exists(_csv+'.0'):
                    print('File exists: Skip xls to csv conversion')
                else:
                    #pdb.set_trace()
                    # Try converting xls to csv through CLI
                    cmd = ['ssconvert', '-S', _file, _csv]
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    o, e = proc.communicate()

                sheet = pd.read_csv(_csv+'.0')

            name = _file.replace(" ", "_").split('/')[-1]
            survey_df_dict[name] = sheet
            sheet = None

        return survey_df_dict

    def get_csv_df_list(self, 
                        _data_files, 
                        **kwargs):
        survey_df_dict = {}
        remove_files_with_phrases = kwargs['ignore_files_with_phrases'] if 'ignore_files_with_phrases' in kwargs.keys() else self.default_args['ignore_files_with_phrases']
        separator = kwargs['separator'] if 'separator' in kwargs.keys() else self.default_args['separator']
        data_files = [_file for _file in _data_files if not any(phrase in _file for phrase in remove_files_with_phrases)]
        
        for _file in data_files:
            sheet = pd.read_csv(_file, 
                                sep=separator,
                                encoding='unicode_escape',
                               low_memory=False)

            name = _file.replace(" ", "_").split('/')[-1]
            survey_df_dict[name] = sheet
            sheet = None

        return survey_df_dict

    def get_spss_df_list(self, 
                        _data_files, 
                        **kwargs):
        return survey_df_dict

    
def get_methods(class_arg):

    assert inspect.isclass(class_arg), 'Expecting a class as an argument'
    
    method_list = []
    # attribute is a string representing the attribute name
    for attribute in dir(class_arg):
        # Get the attribute value
        attribute_value = getattr(class_arg, attribute)
        # Check that it is callable
        if callable(attribute_value):
            # Filter all dunder (__ prefix) methods
            if attribute.startswith('__') == False:
                method_list.append(attribute)

    return method_list


def get_method_fn(method,
                  method_key,
                  methods_list,
                  methods_class):
    
    assert method in methods_list, f'Error: Improper method selected in {method_key} in config.yaml'
    fn_method = getattr(methods_class, method)
    print(f"{method_key}: {fn_method}")
    return fn_method


class SurveyConditioning(object):
    
    def get_nan_idices(sheet):
        '''
        '''
        idx = np.argwhere(pd.isnull(sheet).to_numpy()) #2D indices
        #print(idx)
        return idx
    
    
    def no_nan(_sheet):
        '''
        '''
        sheet = _sheet.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        #print(sheet.shape)
        return sheet

    def substitute_nan(_sheet):
        '''
        '''
        sheet = _sheet.fillna(-1.0)
        #print(sheet.shape)
        return sheet


class PlottingClass:
    MAX_LABEL_LEN = 5
    
    def labels_consistency(self,
                           idx_label):
        return f"{str(idx_label[1])[:self.MAX_LABEL_LEN]}{idx_label[0]}..."
            
    def build_histogram(self,
                        out_dir,
                        parsed_df,
                        settings,
                        survey_name,
                        ignore_label=-1.0):
        #df = pd.DataFrame({'species': ['bear', 'bear', 'marsupial'], 'population': [1864, 22000, 80000]}, index=['panda', 'polar', 'koala'])
        parsed_df = parsed_df.iloc[settings.get("n_discard_columns", 1):, :]
        ignore_labels = settings.get("ignore_labels", [ignore_label])

        ROWS, COLS, SIZE = settings.get("ROWS", 0), settings.get("COLS", 0), settings.get("SIZE", (['8*8', '6*8']))
        assert not (ROWS == 0 and COLS == 0), f'{survey_name} Error: ROWS/COL value not specified in config.yaml'

        SIZE = list(map(lambda x: eval(x), SIZE))
        fig, axes = plt.subplots(figsize=SIZE, dpi=300, nrows=ROWS, ncols=COLS)

        idx = 0
        for name, col in parsed_df.items():
            #assert all(col.map(type) == int) or all(col.map(type) == float)
            sr_count = col.value_counts()
            _ignore_labels = [label for label in ignore_labels if label in sr_count.keys()]
            if len(_ignore_labels) > 0:
                try:
                    sr_count.drop(labels=_ignore_labels, inplace=True)
                except KeyError:
                    pdb.set_trace()

            i, j = idx // COLS, idx % COLS
            if i>= ROWS or j>= COLS:
                pdb.set_trace()
            #print(sr_count.keys().dtype, sr_count.values.dtype)
            ax = axes[i, j]
            
            FONT_SIZE = 5
            if sr_count.sum() > 0:  #Plot the responses in corresponding subplot
                if not (all(isinstance(x, (int,float)) for x in sr_count.keys())):
                    sr_count.index = list(map(self.labels_consistency, [(k_idx, key) for k_idx, key in enumerate(sr_count.keys())]))
                    if len(sr_count.keys()) > 25:
                        FONT_SIZE = 3
                try:
                    ax.bar(sr_count.keys(), sr_count.values, align='center', color='#607c8e')
                    ax.plot(sr_count.keys(), sr_count.values, marker="o", linestyle="", alpha=0.8, color="b")
                    FIRST_TIME = False
                except TypeError:
                    pdb.set_trace()
                
                ax.set_title(f'{name}', fontsize=7, fontweight='bold', color='blue')

                ax.xaxis.set_tick_params(labelsize=FONT_SIZE, labelcolor='r')  #, labelrotation=-60)
                ax.set_xlabel(f'response[{sr_count.sum()}]', fontsize=5, fontweight='bold')
                ax.yaxis.set_tick_params(labelsize=5, labelcolor='r')
                ax.set_ylabel('count', fontsize=5, fontweight='bold')
                ax.grid(axis='y', alpha=0.75)
                #plt.setp(ax.set_xticklabels(sr_count.keys()))
                idx += 1

        if idx < ROWS*COLS:  #Remove excess subplots
            remove_axes = list(range(idx, ROWS*COLS))
            for idx in remove_axes:
                i, j = idx // COLS, idx % COLS
                ax = axes[i, j]
                ax.remove()

        padding = settings.get("padding", 3.0)
        fig.tight_layout(pad=padding)
        plt.savefig(os.path.join(out_dir,f'{survey_name}.png'), bbox_inches='tight')

        fig.clf()
        plt.close(fig)
        plt.close('all')


def recursive_parse_settings(given_setting, modify_setting):
    for key, val in modify_setting.items():
        if isinstance(val, dict):
            recursive_parse_settings(given_setting[key], val)
        else:
            given_setting[key] = val
    return given_setting


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse and plot various surveys')
    parser.add_argument('--SURVEY_INDEX', help='Pick the survey to parse and plot', default=0, type=int)
    cli = parser.parse_args()

    SURVEY_LIST = ['config_echoes.yaml', 'config_briskee.yaml', 'config_cheetah.yaml', 'config_natc.yaml']
    
    with open(SURVEY_LIST[cli.SURVEY_INDEX], 'r') as f:
        args = yaml.safe_load(f)

    data_files = []
    dirs = list(set([__dir if os.path.isdir(os.path.join(args['data_dir'], __dir)) else '' for __dir in os.listdir(args['data_dir'])]))
    for _dir in dirs:
        for filename in os.listdir(os.path.join(args['data_dir'], _dir)):
            if args['format'] == 'rda': # scan for .rdata/.rda
                if filename.endswith(".rda") or filename.endswith(".rdata"): 
                    data_files.append(os.path.join(args['data_dir'], _dir, filename))
                else:
                    continue
            elif args['format'] == 'xls': # scan for .xls/.xlsx
                if filename.endswith(".xls") or filename.endswith(".xlsx"): 
                     data_files.append(os.path.join(args['data_dir'], _dir, filename))
                else:
                    continue
            elif args['format'] == 'csv': # scan for .csv
                if filename.endswith(".csv"): 
                     data_files.append(os.path.join(args['data_dir'], _dir, filename))
                else:
                    continue

            else:
                print('Unsupported file format selcted')

    print(data_files)
    print(len(data_files))

    out_dir = os.path.join(os.getcwd(), args['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(f"{out_dir} directory: exists")


    parser_class_inst = SurveyParser()
    parser_list = get_methods(SurveyParser)
    print(parser_list)
    parser_fn = get_method_fn(f"get_{args['format']}_df_list",
                              args['format'],
                              parser_list,
                              SurveyParser)

    survey_df_dict = parser_fn(parser_class_inst, data_files, **args)  # Example of procedure call with class instance passing


    '''TODO: Handle None dfs'''
    for name, survey_df in survey_df_dict.items():
        print(f"{name}: {survey_df.shape}")
    #print(survey_df)


    # In[5]:

    conditioning_class_inst = SurveyConditioning()
    methods_list = get_methods(SurveyConditioning)
    print(methods_list)


    # In[7]:


    print(args)

    
    try:
        survey_global_settings = args['surveys'].pop('global')
    except "KeyError":
        survey_global_settings = None

    warnings.filterwarnings( "ignore", module = "matplotlib\..*" )


    print(args['surveys'].keys())
    START_FROM = 0
    surveys = dict(itertools.islice(args['surveys'].items(), START_FROM, len(args['surveys'])))

    heap = hpy()
    heap.setref()

    memory_profile = []
    heap_start = heap.heap()

    for key, _val in surveys.items():
        #pdb.set_trace()
        global_settings = deepcopy(survey_global_settings)
        print(colored(f'============================================ {key} ============================================', 'yellow'))
        val = recursive_parse_settings(global_settings, _val) if survey_global_settings is not None else _val
        print(f"survey_global_settings:{survey_global_settings}\n, val:{val}")
        #pdb.set_trace()

        method_fn = get_method_fn(val['conditioning'],
                                  key,
                                  methods_list,
                                  SurveyConditioning)

        matching_df = [survey_df for file_name, survey_df in survey_df_dict.items() if key in file_name]
        assert len(matching_df) > 0, 'Error: Improper survey name in config.yaml'
        parsed_df = method_fn(matching_df[0])  # Example of procedure call without class instance passing
        print(f"before:{matching_df[0].shape}, after:{parsed_df.shape}")

        pltClass = PlottingClass()
        pltClass.MAX_LABEL_LEN = global_settings.get("MAX_LABEL_LEN", 5)

        # Build a histogram of each column in df
        if val['histogram']['enable']:
            pltClass.build_histogram(out_dir, 
                        parsed_df,
                        val['histogram'], 
                        f"{key}_{val['conditioning']}")
        del pltClass, global_settings, val, method_fn, matching_df, parsed_df
        gc.collect

        heap_end = heap.heap()
        curr_heap = (heap_end.size-heap_start.size)//1e3
        memory_profile.append(curr_heap)
        print(colored(f'================= Heap: {curr_heap} MB =================', 'green'))

    fig = plt.figure()
    plt.plot(memory_profile)
    plt.savefig(os.path.join('./','memory_profile.png'), bbox_inches='tight')

    fig.clf()
    plt.close(fig)
    plt.close('all')
