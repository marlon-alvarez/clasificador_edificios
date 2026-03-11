# Manual de Usuario – Clasificador de Edificios Urbanos

## 1. Introducción

Este documento describe cómo utilizar el tablero del **Clasificador de Edificios Urbanos**, una herramienta que permite identificar automáticamente el tipo de edificación presente en una imagen de calle.

El sistema utiliza un modelo de clasificación basado en **ResNet50 fine-tuned**, entrenado para reconocer ocho tipos de edificaciones urbanas.

## 2. Acceso al tablero

El tablero puede accederse a través de la siguiente dirección:

http://54.156.115.19:3000/

## 3. Requisitos para usar la herramienta

Para utilizar el tablero solo se necesita:

- Un navegador web como **Chrome, Firefox, Edge o Safari**.
- Una imagen de una edificación urbana en formato:
  - `.jpg`
  - `.jpeg`
  - `.png`

No se requieren conocimientos técnicos para interactuar con el sistema.

## 4. Interfaz del tablero

La interfaz del tablero está compuesta por tres secciones principales.

### 4.1 Área de carga de imagen

En esta sección el usuario puede subir la imagen que desea clasificar.

**Elementos disponibles:**

- Área de carga o botón **"Subir imagen"**
- Vista previa de la imagen cargada

### 4.2 Botón de clasificación

Una vez cargada la imagen, el usuario puede presionar el botón:

**"Clasificar imagen"**

Este botón envía la imagen a la **API del modelo**, donde se ejecuta el proceso de predicción.

### 4.3 Área de resultados

Después de procesar la imagen, el tablero muestra:

- La **categoría predicha**
- La **probabilidad asociada a cada clase**

Las clases posibles son:

- apartment
- church
- garage
- house
- industrial
- officebuilding
- retail
- roof

## 5. Cómo clasificar una imagen

Siga los siguientes pasos:

**Paso 1**  
Abra el navegador y acceda al tablero utilizando la URL indicada.

**Paso 2**  
En la sección **Subir imagen**, haga clic en el área de carga.

**Paso 3**  
Seleccione una imagen desde su computador que contenga una edificación urbana.

**Paso 4**  
Verifique que la imagen aparezca en la vista previa.

**Paso 5**  
Presione el botón **"Clasificar imagen"**.

**Paso 6**  
Espere unos segundos mientras el modelo procesa la imagen.

**Paso 7**  
Observe el resultado en la sección **Resultado**, donde se mostrará:

- La categoría predicha
- Las probabilidades para cada clase

## 6. Interpretación de resultados

El sistema presenta como resultado principal la **clase con mayor probabilidad**.

Adicionalmente, se muestra un desglose de probabilidades para todas las categorías. Esto permite entender el **nivel de confianza del modelo en la predicción**.

## 7. Recomendaciones de uso

Para obtener mejores resultados:

- Utilice imágenes donde **la edificación sea claramente visible**.
- Evite imágenes con **demasiados objetos que oculten el edificio**.
- Prefiera imágenes **tomadas desde la calle o fachada frontal**.
- Utilice imágenes **bien iluminadas**.

## 8. Ejemplo de uso

Flujo típico de uso del sistema:

1. El usuario abre el tablero en el navegador.
2. Carga una imagen de una edificación.
3. Presiona **Clasificar imagen**.
4. El modelo analiza la imagen.
5. El tablero muestra la predicción y las probabilidades.
