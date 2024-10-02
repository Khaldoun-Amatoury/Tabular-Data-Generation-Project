from sdv.datasets.demo import get_available_demos, download_demo
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
import pandas as pd 
import os 


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
        case _:
            raise ValueError(f"Invalid model name: {model_name}")


# data, metadata = setup()
# print(generate_copulagan_synthesizer(data, metadata, num_rows=100, save_data=True))

#print(get_available_demos(modality='single_table'))
