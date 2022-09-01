curl --location --request POST 'localhost:1234/invocations' \
--header 'Content-Type: application/json' \
--data-raw '{
    "columns":["sepal length (cm)", "sepal width (cm)"],                                          
    "data": [[5.1, 3.5]]          
}'