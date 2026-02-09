# Estymator Czasu Półmaratonu 🏃‍♂️

Aplikacja AI wykorzystująca uczenie maszynowe (PyCaret) i LLM (OpenAI) do przewidywania wyników sportowych.

🚀 **Aplikacja dostępna pod adresem:** (https://sea-lion-app-zhhg9.ondigitalocean.app/)
Notebook z trenowania modelu znajduje się w folderze notebook

### Opis Projektu
Zaimplementowałem aplikację szacującą czas ukończenia półmaratonu...

Zaimplementuj aplikację szacującą czas ukończenia półmaratonu dla zadanych danych

1. Umieść dane w Digital Ocean Spaces

1. Napisz notebook, który będzie Twoim pipelinem do trenowania modelu
    * czyta dane z Digital Ocean Spaces
    * czyści je
    * trenuje model (dobierz odpowiednie metryki [feature selection])
    * nowa wersja modelu jest zapisywana lokalnie i do Digital Ocean Spaces

1. Aplikacja
    * opakuj model w aplikację streamlit
    * wdróż (deploy) aplikację za pomocą Digital Ocean AppPlatform 
    * wejściem jest pole tekstowe, w którym użytkownik się przedstawia, mówi o tym
    jaka jest jego płeć, wiek i czas na 5km
    * jeśli użytkownik podał za mało danych, wyświetl informację o tym jakich danych brakuje
    * za pomocą LLM (OpenAI) wyłuskaj potrzebne dane, potrzebne dla Twojego modelu
    do określenia, do słownika (dictionary lub JSON)
    * tę część podepnij do Langfuse, aby zbierać metryki o skuteczności działania LLM'a
