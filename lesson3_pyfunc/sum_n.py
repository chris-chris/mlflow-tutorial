import pandas as pd
import mlflow.pyfunc


class SumN(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return sum(model_input)


model_path = "sum_n_model"
sum_n_model = SumN()
mlflow.pyfunc.save_model(path=model_path, python_model=sum_n_model)

res = sum_n_model.predict(context=None, model_input=[1, 2, 3])
print(res)


# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_path)

# Evaluate the model
model_input = pd.DataFrame([range(5)])
model_output = loaded_model.predict(model_input)
assert model_output == 10
