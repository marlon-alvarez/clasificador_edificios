# Arquitectura y Despliegue

Sistema de **clasificación de inmuebles** y **asignación de horarios de operación** para Inter Rapidísimo. Compone una API local de orquestación (FastAPI) con un pod GPU en RunPod que sirve dos procesos vLLM (clasificador fine-tuned + base multimodal).

---

## 1. Visión general (componentes)

```mermaid
flowchart LR
    subgraph Client["📱 Cliente"]
        App["Inter App<br/>(Flutter — domiciliario)"]
    end

    subgraph Local["🖥️ API local · FastAPI :8080"]
        direction TB
        EP["POST /processing"]
        ANON["anonymizer.py<br/>OpenCV Haar (CPU)"]
        IMG["images.py<br/>downscale + base64"]
        CLS["classifier.py<br/>cliente OpenAI"]
        EXT["extractor.py<br/>VLM desc. + LLM horario"]
        CONF["config.py<br/>(.env)"]
        EP --> ANON --> IMG
        IMG --> CLS
        IMG --> EXT
        CONF -.-> CLS
        CONF -.-> EXT
    end

    subgraph Pod["☁️ RunPod GPU · vllm/vllm-openai:v0.21.0"]
        direction TB
        VLLM1["vLLM :8000<br/><b>gemma-3-4b-ft</b><br/>(fine-tune · guided_choice)"]
        VLLM2["vLLM :8001<br/><b>gemma-3-4b-it</b><br/>(base multimodal SigLIP)"]
    end

    subgraph HF["🤗 Hugging Face Hub"]
        REPO1["andtorrcan94/gemma-3-4b-ft-merged<br/>(privado)"]
        REPO2["google/gemma-3-4b-it"]
    end

    App -- "multipart<br/>images[]" --> EP
    CLS  -- "OpenAI API · :8000/v1<br/>chat.completions" --> VLLM1
    EXT  -- "OpenAI API · :8001/v1<br/>chat.completions (vision + text)" --> VLLM2

    VLLM1 -. "pull en cold start" .-> REPO1
    VLLM2 -. "pull en cold start" .-> REPO2

    EP -- "JSON consolidado" --> App

    classDef local fill:#E8F4FD,stroke:#2B6CB0,color:#1A365D;
    classDef pod   fill:#FEF5E7,stroke:#B7791F,color:#744210;
    classDef hf    fill:#F3E8FF,stroke:#6B46C1,color:#44337A;
    classDef cli   fill:#E6FFFA,stroke:#2C7A7B,color:#234E52;
    class EP,ANON,IMG,CLS,EXT,CONF local;
    class VLLM1,VLLM2 pod;
    class REPO1,REPO2 hf;
    class App cli;
```

| Capa | Responsabilidad | Tecnología |
|---|---|---|
| **Cliente** | Captura fotos del inmueble en terreno | Flutter (Inter App) |
| **API local** | Orquestación, anonimización, pre/post-proceso | FastAPI + OpenCV + Pillow |
| **Pod GPU** | Inferencia VLM/LLM compatible OpenAI | vLLM 0.21 sobre CUDA 12.9 |
| **HF Hub** | Distribución de pesos (fine-tune mergeado + base) | Hugging Face |

---

## 2. Pipeline de una request `POST /processing`

```mermaid
sequenceDiagram
    autonumber
    participant U as 📱 Inter App
    participant A as 🖥️ FastAPI<br/>(api/main.py)
    participant H as 🧠 OpenCV Haar<br/>(CPU local)
    participant C as 🎯 vLLM :8000<br/>gemma-3-4b-ft
    participant D as 👁️ vLLM :8001<br/>gemma-3-4b-it

    U->>A: multipart images[] (1ª = principal)
    A->>A: load_pil() · validar bytes
    A->>H: anonymize_faces(img) ∀ imagen
    H-->>A: imagen anonimizada + n_caras
    A->>A: downscale_for_vlm() (privacidad antes de salir)

    par Clasificación (1 llamada)
        A->>C: chat.completions (guided_choice)<br/>imagen principal
        C-->>A: label ∈ {casa, apartamento,<br/>local_comercial, unknown}
    and Descripciones (N llamadas)
        A->>D: chat.completions (vision)<br/>prompt main/secondary
        D-->>A: descripción OCR · entorno
    end

    A->>A: concatenar descripciones (" | ")
    A->>D: chat.completions (texto)<br/>parse_schedule()
    D-->>A: JSON horario | null
    A->>A: consolidate_schedule()<br/>(fallback → DEFAULT_SCHEDULE)
    A-->>U: { property_type, classification,<br/>images[], schedule, summary }
```

> **Privacidad first**: la anonimización ocurre **antes** del downscale y **antes** de cualquier salida hacia el pod GPU. Ninguna cara sin difuminar viaja a la red externa.

---

## 3. Despliegue — preparación del modelo (one-shot)

Antes del pod GPU hay un paso *one-shot* que mergea el adapter LoRA con la base y lo publica en HF.

```mermaid
flowchart LR
    subgraph Dev["💻 Local (repo)"]
        ADAPT["deploy/best_adapter/<br/>(LoRA ~2.7 GB)"]
        SCRIPT["merge_lora.py<br/>setup_cpu_pod.sh"]
    end

    subgraph CPU["☁️ RunPod CPU pod (efímero)"]
        direction TB
        ENV["HF_TOKEN · HF_REPO_ID"]
        MERGE["1. cargar base bf16<br/>2. PeftModel.from_pretrained<br/>3. merge_and_unload()<br/>4. save_pretrained (4GB shards)<br/>5. push_to_hub(private)"]
    end

    subgraph HF["🤗 Hugging Face"]
        BASE["google/gemma-3-4b-it"]
        OUT["andtorrcan94/gemma-3-4b-ft-merged"]
    end

    ADAPT  --> CPU
    SCRIPT --> CPU
    ENV    --> MERGE
    BASE   --> MERGE
    MERGE  --> OUT
    OUT    -. "consumido por el pod GPU" .-> Pod[(Pod GPU vLLM)]

    style CPU  fill:#FFFAF0,stroke:#DD6B20
    style HF   fill:#F3E8FF,stroke:#6B46C1
    style Pod  fill:#FEF5E7,stroke:#B7791F
```

**Por qué un pod CPU para el merge:** el merge no requiere GPU (~10–15 min en CPU) y evita pagar GPU mientras solo se copian pesos. El cuello de botella es la descarga del base (~9 GB).

---

## 4. Topología del pod GPU (runtime)

```mermaid
flowchart TB
    subgraph Pod["☁️ Pod RunPod — imagen vllm/vllm-openai:v0.21.0 · CUDA 12.9 · Ubuntu 24.04"]
        direction LR

        subgraph P1["Proceso vLLM #1 · :8000"]
            M1["--model andtorrcan94/gemma-3-4b-ft-merged<br/>--served-model-name gemma-3-4b-ft<br/>--api-key $KEY"]
        end

        subgraph P2["Proceso vLLM #2 · :8001"]
            M2["--model google/gemma-3-4b-it<br/>--served-model-name gemma-3-4b-it<br/>--api-key $KEY"]
        end

        GPU[("🎮 GPU compartida<br/>Ada / Hopper")]
        M1 --- GPU
        M2 --- GPU
    end

    Proxy["🔐 RunPod Proxy<br/>https://<pod>-8000.proxy.runpod.net<br/>https://<pod>-8001.proxy.runpod.net"]
    API["FastAPI local"]

    API <-- "Bearer $RUNPOD_API_KEY" --> Proxy
    Proxy --> P1
    Proxy --> P2
```

| Proceso | Puerto | `--model` | `--served-model-name` | Uso |
|---|---|---|---|---|
| Clasificador (fine-tune) | `8000` | `andtorrcan94/gemma-3-4b-ft-merged` | `gemma-3-4b-ft` | `guided_choice` → 1 palabra exacta |
| Base (multimodal) | `8001` | `google/gemma-3-4b-it` | `gemma-3-4b-it` | Descripción OCR · parse horario |

Ambos procesos comparten **la misma API key** que la FastAPI expone como `RUNPOD_API_KEY` en su `.env`.

---

## 5. Configuración y secretos

```mermaid
flowchart LR
    ENV[".env (api/)<br/>CLASSIFY_URL · DESCRIBE_URL<br/>CLASSIFY_MODEL · DESCRIBE_MODEL<br/>RUNPOD_API_KEY · CLASS_NAMES · PORT"]
    CFG["config.py<br/>load_dotenv(override=True)"]
    CLI1["classifier.py<br/>AsyncOpenAI(base_url=CLASSIFY_URL+/v1)"]
    CLI2["extractor.py<br/>AsyncOpenAI(base_url=DESCRIBE_URL+/v1)"]

    ENV --> CFG --> CLI1
    CFG --> CLI2

    note["⚠️ override=True porque RunPod inyecta<br/>su propia RUNPOD_API_KEY de plataforma"]
    CFG -.-> note
    style note fill:#FFF5F5,stroke:#C53030,color:#742A2A
```

---

## 6. Resumen de flujo end-to-end

```mermaid
flowchart LR
    A["📷 Fotos en terreno"] --> B["📱 Inter App<br/>(Flutter)"]
    B --> C["🖥️ FastAPI<br/>/processing"]
    C --> D["🛡️ Anonimización<br/>local (CPU)"]
    D --> E1["🎯 Clasificador<br/>vLLM :8000"]
    D --> E2["👁️ Descriptor VLM<br/>vLLM :8001"]
    E2 --> F["📅 LLM horario<br/>vLLM :8001"]
    E1 --> G["📦 Respuesta JSON"]
    F --> G
    G --> H["🗺️ Torre de Direcciones<br/>(Inter Rapidísimo)"]

    style A fill:#E6FFFA
    style H fill:#FFFAF0,stroke:#DD6B20,stroke-width:2px
```

> **Resultado de negocio**: enriquece la Torre de Direcciones con tipo de inmueble + horario, reduciendo el 15–20 % de fallos evitables de entrega por horario o tipo incorrecto.
