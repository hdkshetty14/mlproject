from src.pipeline.predict_pipeline import CustomData, PredictPipeline

data = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=72,
    writing_score=74
)

df = data.get_data_as_dataframe()
PredictPipeline().predict(df)
