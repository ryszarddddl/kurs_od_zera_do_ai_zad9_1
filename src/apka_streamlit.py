import streamlit as st
from dotenv import load_dotenv, set_key
import os
import io
from pathlib import Path

load_dotenv() # To wczyta zmienne środowiskowe na serwerze
current_dir = Path(__file__).resolve().parent

import langfuse  # DODAJ TĘ LINIĘ
from langfuse.openai import OpenAI as LangfuseOpenAI
from langfuse import Langfuse
import pandas as pd
import numpy as np
from pycaret.clustering import setup, create_model, assign_model, plot_model, save_model, load_model, predict_model
import joblib
import json
from datetime import datetime
import boto3
import sys
from importlib import metadata as metadata_lib  # Alias dla jasności


# 1. Inicjalizacja stanu sesji (zrób to na początku pliku)
if "last_trace_id" not in st.session_state:
    st.session_state.last_trace_id = None

#@st.cache_data
def handle_api_keys():
    # 1. Próba załadowania z .env
    env_path = Path(".env")
    load_dotenv(env_path)
    
    # Pobranie wartości z systemu/pliku .env
    openai_key = os.getenv("OPENAI_API_KEY")
    lf_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    lf_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    lf_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Sprawdzamy, czy klucze już są załadowane
    keys_loaded = all([openai_key, lf_public_key, lf_secret_key])

    with st.sidebar:
        if not keys_loaded:
            with st.form("api_keys_form"):
                st.subheader("🔑 Konfiguracja API")
                
                openai_input = st.text_input("OpenAI API Key", type="password")
                lf_public_input = st.text_input("Langfuse Public Key")
                lf_secret_input = st.text_input("Langfuse Secret Key", type="password")
                lf_host_input = st.text_input("Langfuse Host", value="https://cloud.langfuse.com")
                
                submit_button = st.form_submit_button("Zatwierdź klucze")
                
                if submit_button:
                    if openai_input and lf_public_input and lf_secret_input:
                        # Ustawienie zmiennych środowiskowych, aby os.getenv je widział po rerun
                        os.environ["OPENAI_API_KEY"] = openai_input
                        os.environ["LANGFUSE_PUBLIC_KEY"] = lf_public_input
                        os.environ["LANGFUSE_SECRET_KEY"] = lf_secret_input
                        os.environ["LANGFUSE_HOST"] = lf_host_input
                        set_key(str(env_path), "OPENAI_API_KEY", openai_input)
                        set_key(str(env_path), "LANGFUSE_PUBLIC_KEY", lf_public_input)
                        set_key(str(env_path), "LANGFUSE_SECRET_KEY", lf_secret_input)
                        set_key(str(env_path), "LANGFUSE_HOST", lf_host_input)
                        st.success("Klucze zapisane!")
                        st.rerun()
                    else:
                        st.error("Wypełnij wszystkie pola!")

    # Blokada ekranu głównego jeśli brak kluczy
    if not (openai_key and lf_public_key and lf_secret_key):
        st.title("🏃 Analiza Półmaratonu")
        st.info("### Skonfiguruj klucze API w panelu bocznym i kliknij przycisk.")
        st.stop()
            
    return openai_key, lf_public_key, lf_secret_key, lf_host

def handle_digital_ocean_keys():
    # 1. Próba załadowania z .env
    current_folder = Path().absolute()
    env_path = current_folder / ".env"
    load_dotenv(env_path)
    #print(env_path)
    # Pobranie wartości z systemu/pliku .env
    do_access_key = os.getenv("DO_ACCESS_KEY")
    do_secret_key = os.getenv("DO_SECRET_KEY")
    do_space_input = os.getenv("DO_SPACE_NAME")
    do_region = os.getenv("DO_REGION")
    do_end_url = os.getenv("DO_END_URL")
    
    # Sprawdzamy, czy klucze już są załadowane
    keys_loaded = all([do_access_key, do_secret_key, do_space_input])

    with st.sidebar:
        if not keys_loaded:
            with st.form("api_keys_form"):
                st.subheader("🔑 Konfiguracja DigitalOcean Spaces")
                do_access_key = st.text_input("Wprowadź DigitalOcean Access Key: ", type="password")
                do_secret_key = st.text_input("Wprowadź secret digital oceans space Key", type="password")
                do_space_input = st.text_input("Wprowadź nazwę Space (Bucketa): ")
                do_region = st.text_input("Wprowadź region: ", value="fra1")
                do_end_url = st.text_input("Wprowadź url region: ", value="https://fra1.digitaloceanspaces.com")

                submit_button = st.form_submit_button("Zatwierdź klucze")
                
                if submit_button:
                    if do_access_key and do_secret_key and do_space_input:
                        # Ustawienie zmiennych środowiskowych, aby os.getenv je widział po rerun
                        os.environ["DO_ACCESS_KEY"] = do_access_key
                        os.environ["DO_SECRET_KEY"] = do_secret_key
                        os.environ["DO_SPACE_NAME"] = do_space_input
                        os.environ["DO_REGION"] = do_region
                        os.environ["DO_END_URL"] = do_end_url
                        set_key(str(env_path), "DO_ACCESS_KEY", do_access_key)
                        set_key(str(env_path), "DO_SECRET_KEY", do_secret_key)
                        set_key(str(env_path), "DO_SPACE_NAME", do_space_input)
                        set_key(str(env_path), "DO_REGION", do_region)
                        set_key(str(env_path), "DO_END_URL", do_end_url)
                        st.success("Klucze zapisane!")
                        st.rerun()
                    else:
                        st.error("Wypełnij wszystkie pola!")
        # Blokada ekranu głównego jeśli brak kluczy
    if not (do_access_key and do_secret_key and do_space_input):
        st.title("🏃 Analiza Półmaratonu")
        st.info("### Skonfiguruj klucze API w panelu bocznym i kliknij przycisk.")
        st.stop() 
   
    return do_access_key, do_secret_key, do_space_input, do_region, do_end_url

def get_data_from_llm(api_key, user_text):
    import os
    import json
    from langfuse.openai import OpenAI
    openai_client = OpenAI(api_key=api_key)

    prompt = f"""Jesteś ekspertem biegowym. Wyodrębnij dane z tekstu użytkownika.
            Zawsze odpowiadaj wyłącznie w formacie JSON z kluczami:
            'Płeć': ('M' lub 'K'),
            'Wiek': (liczba),
            '5 km Czas': ('HH:MM:SS'),
            'Błędy': (lista tekstowa problemów z danymi, np. 'Wiek jest nierealny', 'Brak płci', 'Czas na 5km jest szybszy niż rekord świata').

            Jeśli dane są absurdalne (np. wiek 500 lat, czas 5km 3 minuty), opisz to w 'Błędy'.
            Jeśli danych brakuje, wpisz null w polu danych i dodaj informację do 'Błędy'.

            Dane użytkownika: {user_text}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        name="ekstrakcja-danych-biegacza",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    
    try:
        res = json.loads(raw_content)
        # Wyciągamy błędy (jeśli LLM ich nie wygenerował, dajemy pustą listę)
        errors = res.get('Błędy', [])
        # Czyścimy słownik z klucza 'Błędy', żeby zostały same dane dla modelu
        data = {k: v for k, v in res.items() if k != 'Błędy'}
        return data, errors
    except:
        return {}, ["Błąd krytyczny analizy tekstu."]

def upload_file_to_digital_ocean(local_file_path, access_key, access_secret_key, nazwa_space, remote_path=None, region='fra1', end_url='https://fra1.digitaloceanspaces.com'):
    # Jeśli nie podano nazwy w chmurze, użyj nazwy pliku lokalnego
    if remote_path is None:
        remote_path = os.path.basename(local_file_path)
        
    session = boto3.session.Session()
    client = session.client(
        's3',
        region_name=region.strip(),
        endpoint_url=end_url.strip(),
        aws_access_key_id=access_key.strip(),
        aws_secret_access_key=access_secret_key.strip()
    )
    st.title("Wysyłanie pliku na digital ocean")
    try:
        # Wysyłanie pliku binarnego .pkl
        client.upload_file(local_file_path, nazwa_space, remote_path)
        st.success(f"Plik {local_file_path} został wysłany do Space: {nazwa_space} jako {remote_path}")
    except Exception as e:
        st.error(f"Błąd wysyłki pliku: {e}")

def download_file_from_digital_ocean(local_file_path, access_key, access_secret_key, nazwa_space, remote_path=None, region='fra1', end_url='https://fra1.digitaloceanspaces.com'):
    # Jeśli nie podano nazwy w chmurze, użyj nazwy pliku lokalnego
    if remote_path is None:
        remote_path = os.path.basename(local_file_path)
        
    session = boto3.session.Session()
    client = session.client(
        's3',
        region_name=region.strip(),
        endpoint_url=end_url.strip(),
        aws_access_key_id=access_key.strip(),
        aws_secret_access_key=access_secret_key.strip()
    )
    st.title("Ściąganie pliku z digital ocean")
    try:
        response = client.get_object(Bucket=nazwa_space, Key=remote_path)
        # Zwracamy BytesIO, czyli "wirtualny plik"
        return io.BytesIO(response['Body'].read())
    except Exception as e:
        st.error(f"Błąd pobierania: {e}")
        return None

def list_do_files_by_extension(access_key, access_secret_key, nazwa_space, extension='.csv', region='fra1', end_url='https://fra1.digitaloceanspaces.com'):
    session = boto3.session.Session()
    client = session.client(
        's3',
        region_name=region.strip(),
        endpoint_url=end_url.strip(),
        aws_access_key_id=access_key.strip(),
        aws_secret_access_key=access_secret_key.strip()
    )

    try:
        # Pobranie listy obiektów z Bucketa
        response = client.list_objects_v2(Bucket=nazwa_space)
        #st.title("Lista plików na digital ocean")
        if 'Contents' in response:
            # Filtrowanie plików po rozszerzeniu
            files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith(extension)]
            
            if files:
               # for f in files:
                 #   st.write(f" - {f}")
                return files
            else:
                #st.error(f"Brak plików z rozszerzeniem {extension}.")
                return []
        else:
            #st.error("Bucket jest pusty.")
            return []
            
    except Exception as e:
        #st.error(f"Błąd podczas listowania plików: {e}")
        return []

def send_feedback(value, comment, lf_public, lf_secret, lf_host):
    # Inicjalizacja klienta
    langfuse_client = Langfuse(
        public_key=lf_public,
        secret_key=lf_secret,
        host=lf_host
    )
    
    # W wersji 3.12.0 używamy .create_score zamiast .score
    langfuse_client.create_score(
        trace_id=st.session_state.last_trace_id,
        name="user-feedback",
        value=value,
        comment=comment
    )
    st.toast("Dziękujemy za opinię!", icon="✅")        
            
def make_descriptions(_data_model,new_data,FILE_CLUSTER_NAMES_AND_DESCRIPTIONS,api_key, lf_public, lf_secret,lf_host):
    df_with_clusters  = {}
    with st.status("Przygotowywanie opisów grup...", expanded=True) as status:
        step_placeholder = st.empty()
        step_placeholder.write("⏳ [1/7] Inicjalizacja kluczy")
        import os
        from langfuse.openai import OpenAI
        # Ustawiamy klucze w systemie - to zawsze działa w wersji 3.x
        os.environ["LANGFUSE_PUBLIC_KEY"] = lf_public
        os.environ["LANGFUSE_SECRET_KEY"] = lf_secret
        os.environ["LANGFUSE_HOST"] = lf_host

        # Inicjalizacja bez kłopotliwych argumentów
        openai_client = OpenAI(api_key=api_key)
        
        kmeans_pipeline = _data_model

        step_placeholder.write("⏳ [2/7] Pobranie nazw cech")
        # 2. Pobranie nazw cech (poprzedni błąd)
        # Jeśli model to Pipeline PyCaret, nazwy są w feature_names_in_
        if hasattr(_data_model, 'feature_names_in_'):
            feature_names = list(_data_model.feature_names_in_)
        else:
            feature_names = new_data.select_dtypes(include=['number']).columns.tolist()
        
        step_placeholder.write("⏳ [3/7] Inicjalizacja sesji Pycaret")
        # 1. Inicjalizacja sesji PyCaret (konieczna, aby predict_model widział pipeline)
        # Przekazujemy data_df, aby PyCaret wiedział, na jakich danych operujemy
        setup(
            data=new_data, 
            verbose=False,       # Wyłączamy zbędne komunikaty w konsoli
            html=False,          # Wyłączamy HTML (ważne w Streamlit, by nie psuć widoku)
            session_id=123,       # Stałe ID sesji dla powtarzalności
            preprocess=False 
        )

        step_placeholder.write("⏳ [4/7] Przetwarzanie kolumn w modelu")
        missing_cols = [c for c in feature_names if c not in new_data.columns]
        if missing_cols:
            st.error(f"Model oczekuje kolumn: {missing_cols}, których brakuje w wgranym pliku!")
            # Wyświetl nazwy kolumn w Twoim pliku, żebyś mógł porównać
            st.write("Kolumny w Twoim pliku:", list(new_data.columns))
            return
        
        step_placeholder.write("⏳ [5/7] Predykcja modelu")
        # 3. Teraz predict_model nie wyrzuci błędu NoneType, bo setup() wypełnił 'self.pipeline'
        try:
            step_placeholder.write("⏳ [5/7] Próba 1: Standardowa predykcja PyCaret...")
            df_with_clusters = predict_model(model=_data_model, data=new_data[feature_names])
            
        except Exception as e:
            st.warning(f"⏳ [5/7] Metoda 1 zawiodła: {e}. Próba 2: Metoda przypisania bezpośredniego...")
            try:
                # Metoda 2: Czasami PyCaret wymaga assign_model po setupie
                from pycaret.clustering import assign_model
                df_with_clusters = assign_model(_data_model)
                st.success("Metoda 2 (assign_model) zadziałała!")
                
            except Exception as e2:
                st.warning(f"⏳ [5/7] Metoda 2 zawiodła: {e2}. Próba 3: Bezpośrednie wywołanie silnika modelu...")
                try:
                    # Metoda 3: Wyciągamy surowy model z Pipeline (jeśli to Pipeline)
                    # lub wywołujemy go bezpośrednio, jeśli to obiekt Scikit-Learn
                    if hasattr(_data_model, 'predict'):
                        clusters = _data_model.predict(new_data[feature_names])
                    elif hasattr(_data_model, 'fit_predict'):
                        clusters = _data_model.fit_predict(new_data[feature_names])
                    else:
                        # Ostatnia deska ratunku - sprawdzenie czy to nie jest lista klastrów
                        raise AttributeError("Obiekt nie ma metod predict ani fit_predict")
                    
                    # Ręczne dopisanie klastrów do kopii danych
                    df_with_clusters = new_data.copy()
                    df_with_clusters['Cluster'] = clusters
                    st.success("⏳ [5/7] Metoda 3 (Direct Engine) zadziałała!")
                    
                except Exception as e3:
                    st.error(f"Wszystkie metody zawiodły. Ostatni błąd: {e3}")
                    # Tutaj możesz dodać st.write(type(_data_model)), żeby zobaczyć czym on jest
                    raise e3

        step_placeholder.write("⏳ [6/7] Preprocesing nazw grup")
        cluster_descriptions = {}
        for cluster_id in df_with_clusters['Cluster'].unique():
            cluster_df = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
                
            # --- ZMIANA TUTAJ ---
            # 1. Obliczamy średnie dla kolumn numerycznych (kluczowe statystyki)
            stats = cluster_df.select_dtypes(include=['number']).mean().to_string()
                
            # 2. Pobieramy próbkę 5 losowych wierszy (tylko wybrane kolumny dla kontekstu)
            # Wybierz nazwy kolumn, które faktycznie opisują biegacza
            cols_to_show = ['Miejsce', 'Płeć', 'Kategoria wiekowa', 'Czas', 'Tempo'] 
            # Upewnij się, że te kolumny istnieją w Twoim df, lub użyj: cluster_df.columns[:10]
            sample = cluster_df[cluster_df.columns.intersection(cols_to_show)].sample(min(5, len(cluster_df))).to_string()

            summary = f"Statystyki średnie:\n{stats}\n\nPrzykładowi biegacze:\n{sample}"
            # --- KONIEC ZMIANY ---

            cluster_descriptions[cluster_id] = summary

        step_placeholder.write("⏳ [7/7] Odpytanie openai o nazwy grup")
        prompt = "Użyliśmy algorytmu klastrowania."
        for cluster_id, description in cluster_descriptions.items():
            prompt += f"\n\nKlaster {cluster_id}:\n{description}"

        prompt += """
        Wygeneruj najlepsze nazwy dla każdego z klasterów oraz ich opisy

        Użyj formatu JSON. Przykładowo:
        {
            "Cluster 0": {
                "name": "Klaster 0",
                "description": "W tym klastrze znajdują się osoby, które..."
            },
            "Cluster 1": {
                "name": "Klaster 1",
                "description": "W tym klastrze znajdują się osoby, które..."
            }
        }
        """
        #print(prompt)
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            name="opis-klastrow-maraton", # To zobaczysz w panelu Langfuse
            messages=[{"role": "user", "content": prompt}]
        )

        # Zapisujemy ID, aby łapki wiedziały, którą odpowiedź oceniamy
        st.session_state.last_trace_id = response.id 
        
        status.update(label="Przygotowanie opisów grup zakończone!", state="complete", expanded=False)
    
    result = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    cluster_names_and_descriptions = json.loads(result)
    with open(FILE_CLUSTER_NAMES_AND_DESCRIPTIONS, "w") as f:
        f.write(json.dumps(cluster_names_and_descriptions))
    
def get_cluster_names_and_descriptions(file_path, lf_public, lf_secret, lf_host):
    with open(file_path, "r") as f:
        descriptions = json.load(f)
    
    lf_client = Langfuse(public_key=lf_public, secret_key=lf_secret, host=lf_host)
    
    # W v3.12.0 zamiast .trace() używamy .start_span() dla ręcznych wpisów
    span = lf_client.start_span(
        name="wczytanie-opisow-z-pliku",
        metadata={"file": str(file_path)}
    )
    
    # Zapisujemy ID spanu (które służy jako trace_id dla łapek)
    st.session_state.last_trace_id = span.id
    
    return descriptions

def convert_time_to_seconds(time_val):
    # 1. Jeśli to obiekt czasu ze Streamlit (st.time_input)
    if hasattr(time_val, 'hour'): 
        return time_val.hour * 3600 + time_val.minute * 60 + time_val.second
    
    # 2. Jeśli to liczba (np. ze slidera)
    if isinstance(time_val, (int, float)):
        return int(time_val * 60) # zakładając, że podajesz minuty

    # 3. Jeśli to tekst (z CSV) - Twoja stara logika
    if isinstance(time_val, str):
        if time_val in ['DNS', 'DNF', '']:
            return 0
        parts = time_val.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2: # mm:ss
            return int(parts[0]) * 60 + int(parts[1])
            
    return 0

# KONWERSJA CZASU: sekundy -> HH:MM:SS
def format_seconds(seconds):
    td = pd.to_timedelta(seconds, unit='s')
    # Wyciągamy sam czas bez informacji o dniach
    return str(td).split()[-1].split('.')[0]

#@st.cache_data
def get_model(MODEL_PATH):
    # Wymuś pełną ścieżkę z rozszerzeniem .pkl
    full_path = str(Path(MODEL_PATH).with_suffix('.pkl'))
    return joblib.load(full_path)

@st.cache_data
def get_all_participants(_data_model,new_data):
    # 1. Pobierz DOKŁADNĄ listę kolumn z modelu
    expected_cols = list(_data_model.feature_names_in_)
    
    # 2. Wytnij z dużego DataFrame tylko te kolumny w DOKŁADNEJ kolejności
    # To usunie "Drużynę", "Kraj" i inne, które bolą model
    data = new_data[expected_cols]
    df_with_clusters = predict_model(_data_model, data = data)
    return df_with_clusters

DEBUG_MODE = False
if DEBUG_MODE:
    st.write("--- DIAGNOSTYKA BIBLIOTEK ---")
    st.write(f"Python version: {sys.version}")
    st.write(f"OpenAI version: {openai.__version__} (Path: {openai.__file__})")
    try:
        lf_version = metadata_lib.version("langfuse")
        st.write(f"Langfuse version: {lf_version}")
    except metadata.PackageNotFoundError:
        st.write("Langfuse version: Nie znaleziono paczki!")

    st.write(f"Langfuse path: {langfuse.__file__}")
    st.write("-----------------------------")
    st.write(dir(langfuse.Langfuse)) # To wypisze WSZYSTKIE dostępne metody klasy
    st.write("-----------------------------")

required_keys = ["do_access_key", "do_secret_key", "do_space_input", "do_region", "do_end_url"]
if any(key not in st.session_state for key in required_keys):
    st.session_state.do_access_key, st.session_state.do_secret_key, st.session_state.do_space_input, st.session_state.do_region, st.session_state.do_end_url = handle_digital_ocean_keys()
    st.rerun()

if 'data_df' not in st.session_state:
    st.session_state.data_df = None
if st.session_state.data_df is None:
    lista_csv = [f.name for f in current_dir.glob("*.csv")]
    if lista_csv:
        # 2. Wyświetlamy rozwijaną listę (selectbox)
        wybrany_plik = st.selectbox("Wybierz plik danych do analizy z dysku lokalnego:", lista_csv,index=len(lista_csv) - 1)
    
        # 3. Akcja po wyborze (np. wczytanie ramki danych)
        if st.button("Wczytaj dane z dysku lokalnego"):
            st.session_state.data_df = pd.read_csv(current_dir / wybrany_plik, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
            st.success(f"Pomyślnie wczytano: {wybrany_plik}")
            st.dataframe(st.session_state.data_df.head())
            if st.button("OK"):
                st.rerun()
    else:
        st.warning("⚠️ W folderze aplikacji nie znaleziono żadnych plików CSV.") 
        if st.button("OK"):
            st.rerun()
    
    csv_digi_ocean = list_do_files_by_extension(st.session_state.do_access_key, st.session_state.do_secret_key, st.session_state.do_space_input,'csv',st.session_state.do_region, st.session_state.do_end_url)
    if csv_digi_ocean:
        # 2. Wyświetlamy rozwijaną listę (selectbox)
        wybrany_plik_ocean = st.selectbox("Wybierz plik danych do analizy z digital ocean:", csv_digi_ocean,index=len(csv_digi_ocean) - 1)
    
        # 3. Akcja po wyborze (np. wczytanie ramki danych)
        if st.button("Wczytaj dane z digital ocean"):
            buff = download_file_from_digital_ocean(wybrany_plik_ocean,st.session_state.do_access_key, st.session_state.do_secret_key, st.session_state.do_space_input,None,st.session_state.do_region, st.session_state.do_end_url)
            if buff:
                st.session_state.data_df = pd.read_csv(buff, sep=';', encoding='utf-8-sig')
                st.success(f"Pomyślnie wczytano: {wybrany_plik_ocean}")
                st.dataframe(st.session_state.data_df.head())
            if st.button("OK"):
                st.rerun()
    else:
        st.warning("⚠️ W folderze digital ocean nie znaleziono żadnych plików CSV.") 
        if st.button("OK"):
            st.rerun()
    
else:
    if 'd_model' not in st.session_state:
        st.session_state.d_model = None
    if st.session_state.d_model is None:
        lista_pkl = [f.name for f in current_dir.glob("*.pkl")]
        #if lista_pkl:
        # 2. Wyświetlamy rozwijaną listę (selectbox)
        wybrany_plik = st.selectbox("Wybierz plik modelu treningowego z dysku lokalnego:", lista_pkl,index=len(lista_pkl) - 1)
            
        if st.button("Wczytaj dane"):
            wybrany_plik = wybrany_plik.replace('.pkl', '')
            model_name = Path(wybrany_plik).stem
            target_path = str(current_dir / model_name)
            st.session_state.d_model = get_model(target_path)
            st.success(f"Pomyślnie wczytano: {wybrany_plik}")
            if st.button("OK"):
                st.rerun()
        
        pkl_digi_ocean = list_do_files_by_extension(st.session_state.do_access_key, st.session_state.do_secret_key, st.session_state.do_space_input,'pkl',st.session_state.do_region, st.session_state.do_end_url)
        if pkl_digi_ocean:
            # 2. Wyświetlamy rozwijaną listę (selectbox)
            wybrany_plik_ocean = st.selectbox("Wybierz plik modelu treningowego z digital ocean", pkl_digi_ocean,index=len(pkl_digi_ocean) - 1)
                
            # 3. Akcja po wyborze (np. wczytanie ramki danych)
            if st.button("Wczytaj dane z digital ocean"):
                buff = download_file_from_digital_ocean(wybrany_plik_ocean,st.session_state.do_access_key, st.session_state.do_secret_key, st.session_state.do_space_input,None,st.session_state.do_region, st.session_state.do_end_url)
                if buff:
                    st.session_state.d_model = joblib.load(buff)
                    st.success(f"Pomyślnie wczytano: {wybrany_plik_ocean}")
                    if st.button("OK"):
                        st.rerun()

    else:
        required_keys = ["api_key", "lf_public", "lf_secret", "lf_host"]
        if any(key not in st.session_state for key in required_keys):
            st.session_state.api_key, st.session_state.lf_public, st.session_state.lf_secret,st.session_state.lf_host = handle_api_keys()
            st.rerun()
        else:        
            if 'json_cluster_names_and_descriptions' not in st.session_state:
                st.session_state.json_cluster_names_and_descriptions = None
            if st.session_state.json_cluster_names_and_descriptions is None:
            
                lista_json = [f.name for f in current_dir.glob("*.json")]
                wybrany_plik = st.selectbox("Wybierz plik opisu grup modelu treningowego z dysku lokalnego:", lista_json,index=len(lista_json) - 1)
            
                if st.button("Wczytaj dane"):
                    st.session_state.json_cluster_names_and_descriptions = get_cluster_names_and_descriptions(current_dir / wybrany_plik,st.session_state.lf_public,st.session_state.lf_secret,st.session_state.lf_host)
                    st.success(f"Pomyślnie wczytano: {wybrany_plik}")
                    if st.button("OK"):
                        st.rerun()
                
                json_digi_ocean = list_do_files_by_extension(st.session_state.do_access_key, st.session_state.do_secret_key, st.session_state.do_space_input,'json',st.session_state.do_region, st.session_state.do_end_url)
                if json_digi_ocean:
                    # 2. Wyświetlamy rozwijaną listę (selectbox)
                    wybrany_plik_ocean = st.selectbox("Wybierz plik opisu grup modelu treningowego z digital ocean", json_digi_ocean,index=len(json_digi_ocean) - 1)
                
                    # 3. Akcja po wyborze (np. wczytanie ramki danych)
                    if st.button("Wczytaj dane z digital ocean"):
                        buff = download_file_from_digital_ocean(wybrany_plik_ocean,st.session_state.do_access_key, st.session_state.do_secret_key, st.session_state.do_space_input,None,st.session_state.do_region, st.session_state.do_end_url)
                        if buff:
                            st.session_state.json_cluster_names_and_descriptions = json.load(buff)
                            st.success(f"Pomyślnie wczytano: {wybrany_plik_ocean}")
                        if st.button("OK"):
                            st.rerun()
                else:
                    st.warning("⚠️ W folderze digital ocean nie znaleziono żadnych plików JSON.") 
                    if st.button("OK"):
                        st.rerun()

                with st.form("Wygeneruj opisy dla modelu treningowego"):
                    CLUSTER_NAMES_AND_DESCRIPTIONS = st.text_input("Nazwa modelu:", value='half_maraton_cluster_names_and_descriptions_v1.json')
                    submit_button = st.form_submit_button("Zatwierdź nazwę opisu modelu")
                    if not submit_button:
                        st.stop()
                    make_descriptions(st.session_state.d_model,st.session_state.data_df,CLUSTER_NAMES_AND_DESCRIPTIONS,st.session_state.api_key,st.session_state.lf_public,st.session_state.lf_secret,st.session_state.lf_host)
                    st.write('Skończyłem generowanie opisów do modelu treningowego')
                    st.session_state.json_cluster_names_and_descriptions = get_cluster_names_and_descriptions(CLUSTER_NAMES_AND_DESCRIPTIONS,st.session_state.lf_public,st.session_state.lf_secret,st.session_state.lf_host)
                            
                    
                    upload_file_to_digital_ocean(CLUSTER_NAMES_AND_DESCRIPTIONS,st.session_state.do_access_key, st.session_state.do_secret_key, st.session_state.do_space_input,None,st.session_state.do_region, st.session_state.do_end_url)
                if st.button("OK"):
                    st.rerun()
            else:               
                # Tworzymy sidebar
                with st.sidebar:
                    st.header("🤖 Analiza LLM") # Opcjonalny nagłówek
                    
                    user_input = st.text_area(
                        "Przedstaw się i opisz swój start (wiek, płeć, czas na 5km):",
                        value="Nazywam się Jan, mam 20 lat, jestem mężczyzną, 5km biegam w 30 min",
                        height=150
                    )
                    
                    if st.button("Analizuj"):
                        if not user_input.strip():
                            st.warning("Wpisz najpierw tekst!")
                        else:
                            with st.spinner("LLM sprawdza Twoje dane..."):
                                dane, bledy = get_data_from_llm(st.session_state.api_key, user_input)   
                                
                                if bledy:
                                    st.error("Znaleziono problemy w Twoim opisie:")
                                    for bład in bledy:
                                        st.write(f"🚩 {bład}")
                                        # KLUCZOWE: Czyścimy flagę i dane, aby stary profil zniknął
                                    st.session_state.data_ready = False
                                    st.session_state.dane = None
                                    st.info("Popraw swój opis, aby był bardziej realistyczny.")
                                else:
                                    st.success("Dane zweryfikowane pomyślnie!")
                                    st.session_state.dane = dane
                                    st.session_state.data_ready = True

                
                if DEBUG_MODE: 
                    st.write(f"Wybrano: {st.session_state.dane['5 km Czas']} minut")
                
                if 'data_ready' not in st.session_state or not st.session_state.data_ready:
                    st.info("👈 Proszę wprowadzić dane w panelu bocznym i kliknąć 'Analizuj', aby kontynuować.")
                    st.stop() # TUTAJ kod się zatrzymuje
                else:
                    # Poprawka w tworzeniu input_data
                    st.session_state.cols =[]
                    st.session_state.cols = st.session_state.d_model.feature_names_in_               
                    input_data = {}

                    sekundy_5km = convert_time_to_seconds(st.session_state.dane['5 km Czas'])
                    tempo_min_km = sekundy_5km / 5 / 60 # np. 5.3 min/km

                    # Zamiast zer, wypełnij dane szacunkowe na podstawie tempa z 5km
                    for col in st.session_state.cols:
                        if '5 km Czas' in col: input_data[col] = float(sekundy_5km)
                        elif '10 km Czas' in col: input_data[col] = float(sekundy_5km * 2.0)
                        elif '15 km Czas' in col: input_data[col] = float(sekundy_5km * 3.0)
                        elif '20 km Czas' in col: input_data[col] = float(sekundy_5km * 4.0)
                        elif 'Czas' in col: input_data[col] = float(sekundy_5km * 4.22) # estymacja półmaratonu
                        elif 'Tempo' in col and 'Stabilność' not in col: input_data[col] = float(tempo_min_km)
                        elif 'Rocznik' in col: input_data[col] = datetime.now().year - int(st.session_state.dane['Wiek'])
                        elif 'Rok_maratonu' in col: input_data[col] = int(datetime.now().year)
                        elif 'Miejsce'in col or 'Numer startowy' in col or 'Płeć Miejsce' in col or 'Kategoria wiekowa Miejsce' in col or '5 km Miejsce Open' in col or '10 km Miejsce Open' in col or '15 km Miejsce Open' in col or '20 km Miejsce Open' in col: 
                            input_data[col] = int(st.session_state.data_df[col].mode()[0])
                        elif 'Płeć' in col: input_data[col] =  str(gender)
                        elif 'Plec_K' in col: 
                            if st.session_state.dane['Płeć'] =='K':
                                input_data[col] =  1
                            else:
                                input_data[col] =  0
                        elif 'Plec_M' in col:
                            if st.session_state.dane['Płeć']=='M':
                                input_data[col] =  1
                            else:
                                input_data[col] =  0
                        elif 'Kategoria wiekowa' in col: input_data[col] = str(f"{st.session_state.dane['Płeć']}{(dane['Wiek'])//10 *10}")
                        elif 'Grupa wiekowa' in col: input_data[col] = str((dane['Wiek'])//10 *10)
                        else:
                            if pd.api.types.is_numeric_dtype(st.session_state.data_df[col]):
                                input_data[col] = float(st.session_state.data_df[col].mean())
                            else:
                                # Dla tekstu bierzemy najczęstszą wartość (mode) zamiast średniej
                                input_data[col] = st.session_state.data_df[col].mode()[0]
                    
                    person_df = pd.DataFrame([input_data])

                    if DEBUG_MODE:
                        st.write("--- TEST TYPÓW W SŁOWNIKU ---")
                        for klucz, wartosc in input_data.items():
                            st.write(f"Kolumna: {klucz} | Wartość: {wartosc} | Typ: {type(wartosc)}")
                        st.write("-----------------------------")
                    # Wyświetlenie typów danych w tabeli diagnostycznej
                        st.write("### Diagnostyka typów kolumn w person_df")
                        st.write(person_df.dtypes)

                        # Automatyczne znalezienie kolumn, które NIE są numeryczne
                        non_numeric_cols = person_df.select_dtypes(exclude=['number']).columns.tolist()

                        if non_numeric_cols:
                            st.error(f"Znaleziono kolumny tekstowe, które psują model: {non_numeric_cols}")
                        else:
                            st.success("Wszystkie kolumny w person_df są numeryczne!")
                        
                        st.write(f"DEBUG: Obliczone sekundy: {sekundy}")
                        st.write(person_df)

                    all_df = get_all_participants(st.session_state.d_model,st.session_state.data_df)
                    predicted_cluster_id = predict_model(st.session_state.d_model, data=person_df)["Cluster"].values[0]
                    same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
                    descriptions = st.session_state.json_cluster_names_and_descriptions
                    cluster_data = descriptions.get(predicted_cluster_id, {
                        "name": f"Grupa {predicted_cluster_id}",
                        "description": "Brak szczegółowego opisu dla tego klastra. Model wygenerował dużą liczbę niszowych grup."
                    })

                    # 1. Oblicz statystyki dla klastra (średnie wartości)
                    cluster_stats = same_cluster_df.select_dtypes(include=['number']).mean().to_frame().T

                    # 2. Wyświetlenie wyniku w aplikacji
                    st.success(f"Należysz do klastra: **{cluster_data['name']}**", icon="🏃‍♂️")
                    st.info(cluster_data['description'])

                    # 3. Zamiast całego DataFrame, pokaż podsumowanie i małą próbkę
                    with st.expander("📊 Zobacz statystyki Twojej grupy"):
                        st.write("Średnie wyniki biegaczy w Twoim klastrze:")
                        st.dataframe(cluster_stats)
        
                        # Pokazujemy tylko 5 losowych osób dla porównania
                        # 1. Zdefiniuj listę kolumn, które faktycznie interesują biegacza
                        widoczne_kolumny = [
                            'Miejsce', 
                            'Czas', 
                            'Tempo', 
                            'Płeć', 
                            'Kategoria wiekowa', 
                            'Rocznik', 
                            'Rok_maratonu'
                        ]

                        # 2. Sprawdź, które z tych kolumn faktycznie istnieją w Twoim DataFrame
                        dostepne_kolumny = [c for c in widoczne_kolumny if c in same_cluster_df.columns]

                        # 3. Wyświetl tylko przefiltrowaną próbkę
                        st.write("Przykładowi biegacze z Twojego klastra (podobne wyniki):")
                        st.dataframe(
                            same_cluster_df[dostepne_kolumny].sample(min(5, len(same_cluster_df))),
                            use_container_width=True
                        )

                        # Obliczenia
                        total_participants = len(all_df)
                        cluster_count = len(same_cluster_df)
                        cluster_percentage = (cluster_count / total_participants) * 100

                        # Wyświetlanie wskaźników
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Liczba osób w Twoim klastrze", f"{cluster_count}")
                        with col2:
                            st.metric("Procent wszystkich uczestników", f"{cluster_percentage:.1f}%")

                        st.subheader("🚀 Jak trafić do innych grup?")

                        # 1. Obliczamy średnie dla klastrów
                        all_clusters_summary = all_df.groupby("Cluster")[["Tempo", "Czas", "5 km Czas","Rocznik"]].mean()

                        # 2. Dodajemy nazwy klastrów z Twojego JSONa
                        all_clusters_summary['Nazwa klastra'] = all_clusters_summary.index.map(
                            lambda x: st.session_state.json_cluster_names_and_descriptions.get(x, {}).get('name', f"Cluster {x}")
                        )

                        # 3. FORMATOWANIE: Rocznik na liczbę całkowitą (int) i zaokrąglenie tempa
                        all_clusters_summary['Rocznik'] = all_clusters_summary['Rocznik'].astype(int)
                        all_clusters_summary['Tempo'] = all_clusters_summary['Tempo'].round(2)

                        all_clusters_summary['5 km Czas'] = all_clusters_summary['5 km Czas'].round(0).astype(int)

                        all_clusters_summary['5 km Czas'] = all_clusters_summary['5 km Czas'].apply(format_seconds)
                        all_clusters_summary['Czas'] = all_clusters_summary['Czas'].round(0).astype(int)

                        all_clusters_summary['Czas'] = all_clusters_summary['Czas'].apply(format_seconds)

                        # 4. CZYSZCZENIE I WYŚWIETLANIE
                        display_df = all_clusters_summary.reset_index(drop=True)
                        display_df = display_df[['Nazwa klastra', 'Tempo', '5 km Czas','Czas', 'Rocznik']]

                        st.write("Porównaj swoje dane ze średnimi wynikami innych grup:")
                        st.dataframe(display_df, use_container_width=True)

                    # Sekcja opinii (łapki)
                    api_key, lf_public, lf_secret, lf_host = handle_api_keys()
                    # ... (dalej kod z opiniami)

                    # Przycisk otwierający formularz opinii
                    with st.popover("⭐ Oceń ten opis"):
                        st.write("Jak bardzo opis do Ciebie pasuje? (0 - wcale, 5 - idealnie)")
        
                    # Suwak lub feedback (gwiazdki)
                    stars = st.feedback("stars") # W 2026 st.feedback zwraca int 0-4, dodajemy 1
        
                    comment = st.text_area("Twoje uwagi (opcjonalnie):", placeholder="Np. Wszystko się zgadza, ale wiek jest zawyżony...")
        
                    if st.button("Wyślij opinię"):
                        if stars is not None:
                            # Przesunięcie skali z 0-4 na 1-5
                            final_score = stars + 1 
                            send_feedback(final_score, comment, lf_public, lf_secret, lf_host)
                        else:
                            st.warning("Zaznacz gwiazdki przed wysłaniem!")


