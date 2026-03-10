# Clasificador de Edificios

Sistema de clasificacion automatica de edificios urbanos mediante Deep Learning.

---

## Contenido

- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalacion y ejecucion en local](#instalacion-y-ejecucion-en-local)
  - [Backend (FastAPI)](#backend-fastapi)
  - [Frontend (Next.js)](#frontend-nextjs)
- [Despliegue en produccion](#despliegue-en-produccion)
  - [Backend en produccion](#backend-en-produccion)
  - [Frontend en produccion con PM2](#frontend-en-produccion-con-pm2)
- [Verificacion rapida](#verificacion-rapida)
- [Comandos utiles](#comandos-utiles)
- [Troubleshooting](#troubleshooting)

---

## Arquitectura

- `backend`: API en FastAPI que carga el modelo y expone el endpoint `/predict`.
- `frontend`: aplicacion en Next.js para cargar imagenes y mostrar prediccion.

Flujo general:
1. El frontend envia una imagen al backend.
2. El backend preprocesa la imagen y ejecuta inferencia.
3. El backend responde con clase, etiqueta, confianza y probabilidades.

---

## Requisitos

Antes de iniciar, instala:

- Python `3.11.12`
- Node.js `22` y `npm`
- PM2 (solo para frontend en produccion):

```bash
npm install -g pm2
```

---

## Instalacion y ejecucion en local

### Backend (FastAPI)

Desde la raiz del proyecto:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API local: `http://localhost:8000`

Notas:
- En el primer arranque, el backend descarga el modelo desde Google Drive.
- Si cierras la terminal, recuerda activar de nuevo el entorno: `source .venv/bin/activate`.

### Frontend (Next.js)

Desde la raiz del proyecto:

```bash
cd frontend
npm install
npm run dev
```

Frontend local: `http://localhost:3000`

---

## Despliegue en produccion

### Backend en produccion

Para backend en produccion se mantiene el mismo flujo: crear `.venv`, instalar y correr.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend en produccion con PM2

Desde la raiz del proyecto:

```bash
cd frontend
npm install
npm run build
pm2 start npm --name clasificador-frontend -- start
```

Comandos principales de PM2:

```bash
pm2 status
pm2 logs clasificador-frontend
pm2 restart clasificador-frontend
pm2 stop clasificador-frontend
pm2 delete clasificador-frontend
pm2 save
```

Sugerencia para reinicio automatico tras reboot:

```bash
pm2 startup
```

---

## Verificacion rapida

- Backend activo: abre `http://localhost:8000/docs`
- Frontend activo: abre `http://localhost:3000`
- Proceso de frontend en PM2: `pm2 status`

---

## Comandos utiles

### Backend

```bash
# activar entorno virtual
source backend/.venv/bin/activate

# correr backend en modo desarrollo
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
# desarrollo
npm run dev

# produccion
npm run build
npm run start
```

---

## Troubleshooting

- Error `python3: command not found`:
  - Instala Python `3.11.12` y vuelve a intentar.
- Error al activar `.venv`:
  - Verifica que existe `backend/.venv`.
  - Si no existe, recrealo con `python3 -m venv .venv`.
- Error de dependencias en backend:
  - Ejecuta `pip install --upgrade pip` y luego `pip install -r requirements.txt`.
- PM2 no reconocido:
  - Ejecuta `npm install -g pm2`.
- El frontend no levanta en produccion:
  - Asegurate de correr primero `npm run build` antes de `pm2 start ...`.
