from sdv.datasets.demo import get_available_demos, download_demo
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
import pandas as pd 
import os 
# from models.ctabgan_model.ctabgan import CTABGAN
from ctabgan_model.ctabgan import CTABGAN
# from models.ctabgan_model_plus.ctabgan import CTABGAN as CTABGANPLUS
from ctabgan_model_plus.ctabgan import CTABGAN as CTABGANPLUS
from be_great import GReaT
from sklearn.datasets import fetch_california_housing


def get_available_datasets():
    return get_available_demos(modality='single_table')


def setup():

    data, metadata = download_demo(
        modality='single_table',
        dataset_name='adult'
    )

    return (data, metadata)

def generate_gaussian_copula_synthesizer(data, metadata, num_rows=10, save_data=False):
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows)
    if save_data:
        save_path = os.path.join(os.getcwd(), 'datasets', 'synthetic_data_guassian.csv')
        synthetic_data.to_csv(save_path, index=False)
    return synthetic_data


def generate_ctgan_synthesizer(data, metadata, num_rows=10, save_data=False):
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows)
    if save_data:
        save_path = os.path.join(os.getcwd(), 'datasets', 'synthetic_data_ctgan.csv')
        synthetic_data.to_csv(save_path, index=False)
    return synthetic_data


def generate_tvaesynthesizer(data, metadata, num_rows=10, save_data=False):
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows)
    if save_data:
        save_path = os.path.join(os.getcwd(), 'datasets', 'synthetic_data_tvae.csv')
        synthetic_data.to_csv(save_path, index=False)
    return synthetic_data


def generate_copulagan_synthesizer(data, metadata, num_rows=10, save_data=False):
    synthesizer = CopulaGANSynthesizer(metadata)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows)
    if save_data:
        save_path = os.path.join(os.getcwd(), 'datasets', 'synthetic_data_copulagan.csv')
        synthetic_data.to_csv(save_path, index=False)
    return synthetic_data


def ctabgan_synthesizer(data, save_data=False):
    synthesizer =  CTABGAN(raw_csv_path = data,
                 test_ratio = 0.20,  
                 categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                                        'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]}, 
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": 'income'},
                 epochs = 50)

    synthesizer.fit()
    synthetic_data = synthesizer.generate_samples()
    if save_data:
        save_path = os.path.join(os.getcwd(), 'datasets', 'synthetic_data_ctabgan.csv')
        synthetic_data.to_csv(save_path, index=False)

    return synthetic_data


def generate_ctabgan_plus_synthesizer(data, save_data=False):
    synthesizer =  CTABGANPLUS(raw_csv_path = data,
                    test_ratio = 0.20,
                    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                    log_columns = [],
                    mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                    general_columns = ["age"],
                    non_categorical_columns = [],
                    integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                    problem_type= {"Classification": 'income'}) 

    synthesizer.fit()
    syn = synthesizer.generate_samples()
    if save_data:
        save_path = os.path.join(os.getcwd(), 'datasets', 'synthetic_data_ctabgan_plus.csv')
        syn.to_csv(save_path, index=False)
    return syn


def generate_great_synthesizer(data,  num_rows=10, save_data=False):
    synthesizer = GReaT(llm='distilgpt2', batch_size=32,  epochs=50, fp16=True)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows)
    if save_data:
        save_path = os.path.join(os.getcwd(), 'datasets', 'synthetic_data_great.csv')
        synthetic_data.to_csv(save_path, index=False)
    return synthetic_data


def generate_data(model_name, data, metadata, num_rows=10, save_data=False):
    match model_name.lower():
        case 'gaussian_copula':
            return generate_gaussian_copula_synthesizer(data, metadata, num_rows=num_rows, save_data=save_data)
        case 'ctgan':
            return generate_ctgan_synthesizer(data, metadata, num_rows=num_rows, save_data=save_data)
        case 'tvaesynthesizer':
            return generate_tvaesynthesizer(data, metadata, num_rows=num_rows,save_data=save_data)
        case 'copulagan':
            return generate_copulagan_synthesizer(data, metadata, num_rows=num_rows, save_data=save_data)
        case "great":
            return generate_great_synthesizer(data, num_rows=num_rows, save_data=save_data)
        case "ctabganplus":
            return generate_ctabgan_plus_synthesizer(data, save_data=save_data)
        case _:
            raise ValueError(f"Invalid model name: {model_name}")


def get_available_model_names():
    return ["gaussian_copula", "ctgan", "tvaesynthesizer", "copulagan", "great", "ctabganplus"]

data, metadata = setup()
# print(generate_copulagan_synthesizer(data, metadata, num_rows=100, save_data=True))

#print(    get_available_demos(modality='single_table'))
#data =     get_available_demos(modality='single_table')
# print(generate_ctabgan_plus_synthesizer(data='/home/techoffice/code/sdv-project/datasets/Adult.csv'))
print(generate_great_synthesizer(data))


#print( data.iloc[0])