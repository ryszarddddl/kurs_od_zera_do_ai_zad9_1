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
  
4. Instalacja i uruchomienie
Aby zainstalować aplikację, uruchom konsolę (CMD w Windows, Terminal w Linux/macOS) i postępuj zgodnie z wybraną metodą:
Krok 1: Pobranie projektu na dysk
Wybierz jedną z opcji:

    Git: git clone https://github.com/ryszarddddl/Estymator_Czasu_Pol_maratonu
    Wget: wget https://github.com
    Manualnie: Kliknij zielony przycisk "Code" na górze strony i wybierz "Download ZIP" lub "Open with GitHub Desktop".

Krok 2: Instalacja i start
Opcja A: Docker (Zalecane)

    Zainstaluj Docker Desktop: https://docs.docker.com/desktop/
    W konsoli przejdź do folderu projektu: cd Estymator_Czasu_Pol_maratonu
    Zbuduj obraz: docker build -t estymator_maratonu .
    Uruchomienie:
        Otwórz Docker Desktop, wejdź w zakładkę Images i kliknij Run przy estymator_maratonu.
        W ustawieniach (Optional settings) wpisz port (np. 8501).
        Adres do aplikacji znajdziesz w zakładce Containers. W razie problemów zapytaj Gordona (AI wbudowane w Docker Desktop).

Opcja B: Instalacja manualna

    Pobierz Pythona (zalecana wersja 3.11): https://www.python.org/downloads/
    W konsoli zainstaluj wymagane biblioteki:
 
    pip install --upgrade pip
    pip install -r requirements.txt

    Używaj kodu z rozwagą.
    Uruchomienie: Wpisz w konsoli:
   
    streamlit run src/apka_streamlit.py
      
