import sdv


from sdv.datasets.demo import get_available_demos

dataset = get_available_demos(modality='single_table')

print(dataset)

from sdv.datasets.demo import download_demo

data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests'
)

print(data.head())
print(metadata)
# print(metadata.get_column_names(sdtype='numerical'))
# print(metadata.validate())

# metadata.update_column(
#     column_name='amenities_fee',
#     sdtype='categorical')

# metadata.add_column(
#   column_name='cell_numbers',
#   sdtype='phone_number',
#   pii=True
# )


print(metadata)

anonymized_metadata = metadata.anonymize()
print(anonymized_metadata)

# metadata_vis = metadata.visualize(
#     show_table_details='summarized',
# )
# print(metadata_vis)

# from sdv.metadata import SingleTableMetadata

# metadata = SingleTableMetadata()
# metadata_data = metadata.detect_from_dataframe(data)

from sdv.single_table import GaussianCopulaSynthesizer

synthesizer = GaussianCopulaSynthesizer(metadata)

# synthesizer = GaussianCopulaSynthesizer(
#     metadata, # required
#     enforce_min_max_values=True,
#     enforce_rounding=False,
#     numerical_distributions={
#         'amenities_fee': 'beta',
#         'checkin_date': 'uniform'
#     },
#     default_distribution='norm'
# )

synthesizer.fit(data)

synthetic_data = synthesizer.sample(num_rows=10)

print(synthetic_data)

from sdv.single_table import CTGANSynthesizer

# synthesizer = CTGANSynthesizer(metadata)

synthesizer = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=False,
    epochs=500,
    verbose=True
)

synthesizer.fit(data)
synthesizer.get_loss_values()
synthetic_data = synthesizer.sample(num_rows=10)
