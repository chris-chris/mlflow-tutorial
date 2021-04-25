curl --location --request POST 'localhost:1234/invocations' \
--header 'Content-Type: application/json' \
--data-raw '{
    "columns":["sepal length (cm)", "sepal width (cm)", "petal length (cm)",  "petal width (cm)"],
    "data": [[5.1, 3.5, 1.4, 0.2]]
}'

