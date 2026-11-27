#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PI8 · T6 · Elvis Max Estela Paucar · UNMSM · Grupo A · 2026-II.

CONSTRUCCIÓN INICIAL PARA EL PARCIAL
Compara exactamente dos modelos:
  1. Línea base: TF-IDF (representación del texto) + regresión logística.
  2. BETO: modelo preentrenado en español ajustado con nuestros comentarios.

PREPARACIÓN
Usar Python 3.12 y las dependencias del requirements.txt entregado:
  python -m pip install -r requirements.txt
Colocar este .py y Dataset_atencion_1000_revisado.xlsx en la misma carpeta.

EJECUCIÓN PARA EL VÍDEO
  python PI8_T6_EstelaMax_Construccion.py
Ahora el modo predeterminado es ambos: línea base y BETO.
También es válido indicar explícitamente --modelo ambos.
Para una comprobación rápida solo de la línea base:
  python PI8_T6_EstelaMax_Construccion.py --modelo baseline
Para clasificar además un comentario de demostración:
  python PI8_T6_EstelaMax_Construccion.py --predecir "Me atendieron con paciencia"
Para verificar reglas básicas sin entrenar:
  python PI8_T6_EstelaMax_Construccion.py --autoprueba
Para consultar opciones: añadir --help.

MAPA DEL CÓDIGO PARA EXPLICARLO
C1 cargar_validar: leer el corpus, revisar etiquetas y conservar campos mínimos.
C2 agrupar_partir: separar textos similares y obtener train, validación y test.
C3 entrenar_baseline / entrenar_beto: aprender con train y elegir con validación.
C4 metricas / comparar: medir los aciertos sobre el mismo test para ambos.
C5 main / exportar: coordinar los pasos y guardar evidencias y reporte HTML.

Con este Excel y semilla 42: 600 registros de entrenamiento, 200 de validación
y 200 de prueba. Validación NO ajusta pesos: selecciona C o la mejor época.
Test NO selecciona modelos: se usa para la evaluación final.
Cada corrida crea una carpeta nueva en resultados_PI8. Abrir su reporte.html.
Las carpetas de corridas anteriores no se modifican.

CORPUS Y ALCANCE
Fuente inmediata: Excel suministrado para el proyecto; hoja Dataset y hoja Guía.
1.000 registros sintéticos; etiquetas de referencia sin validación humana.
No es una extracción de una entidad de salud ni acredita permisos institucionales.
Autoría de generación y condiciones de redistribución: por documentar.
Criterio: negativo=0, neutral=1, positivo=2 según la referencia del Excel.
El código comprueba coherencia de códigos, pero no verifica la verdad de la etiqueta.
Estos resultados son una prueba técnica, no evidencia de satisfacción real.

BETO descarga pesos públicos la primera vez; el texto se procesa localmente.
--checkpoint admite una carpeta local; --datos permite otra ruta al corpus.
Con PyTorch CPU usar --dispositivo auto o cpu. CUDA exige instalación compatible.
Las métricas se calculan, nunca se fijan para favorecer un modelo.
Referencias técnicas:
https://github.com/dccuchile/beto
https://scikit-learn.org/stable/common_pitfalls.html
https://huggingface.co/docs/transformers/tasks/sequence_classification
"""
from __future__ import annotations
import argparse, hashlib, html, importlib.metadata, json, logging, platform
import re, sys, time, unicodedata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import binomtest
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Etiquetas y orden fijo de las clases en todas las matrices.
CLASES = {0: "negativo", 1: "neutral", 2: "positivo"}
CODIGOS = {v: k for k, v in CLASES.items()}
CHECKPOINT = "dccuchile/bert-base-spanish-wwm-uncased"
# Fijar la revisión de los pesos evita cambiar de base preentrenada entre corridas.
REVISION = "d1c9c4565c9d6731e57ed7f027b802697bad861e"


def json_out(path, obj):
    """Guardar un resultado en JSON legible; conservar tildes y caracteres españoles."""
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def normalizar(s):
    """Unificar Unicode y espacios sin borrar negaciones, acentos ni puntuación."""
    return re.sub("\\s+", " ", unicodedata.normalize("NFKC", str(s))).strip()


def cargar_validar(path):
    """C1 · Leer y validar el corpus antes de permitir cualquier entrenamiento.

    Devuelve:
        limpio: tabla con ID, texto, etiqueta numérica y origen.
        calidad: controles de integridad y huella SHA-256 del archivo.
    Solo el texto se utilizará como característica. El código de sentimiento es
    la respuesta de referencia, por lo que introducirlo como entrada sería fuga
    de información. Los nombres y médicos se excluyen del corpus preparado.
    """
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        d = pd.read_excel(path, sheet_name="Dataset")
    elif path.suffix.lower() == ".csv":
        d = pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError("Use Excel .xlsx o CSV UTF-8.")
    # Estas columnas son obligatorias; las demás no entran al clasificador.
    req = [
        "id_atencion",
        "comentario_atencion",
        "etiqueta_sentimiento",
        "codigo_sentimiento",
        "origen_dato",
    ]
    if set(req) - set(d):
        raise ValueError("Faltan columnas: " + str(sorted(set(req) - set(d))))
    if d[req].isna().any().any():
        raise ValueError("Nulos en columnas obligatorias.")
    for c in [x for x in req if x != "codigo_sentimiento"]:
        if not d[c].map(lambda x: isinstance(x, str)).all():
            raise ValueError(c + " debe contener texto.")
        d[c] = d[c].map(normalizar)
        if d[c].eq("").any():
            raise ValueError("Cadenas vacías en " + c)
    d["etiqueta_sentimiento"] = d.etiqueta_sentimiento.str.lower()
    if not d.etiqueta_sentimiento.isin(CODIGOS).all():
        raise ValueError("Etiqueta desconocida.")
    # Verificar que la etiqueta escrita coincide con su código numérico.
    y = pd.to_numeric(d.codigo_sentimiento, errors="coerce")
    if not y.eq(d.etiqueta_sentimiento.map(CODIGOS)).all():
        raise ValueError("Código y etiqueta incoherentes.")
    if d.id_atencion.duplicated().any():
        raise ValueError("Identificadores repetidos.")
    if not d.origen_dato.str.lower().isin(["sintético", "sintetico"]).all():
        raise ValueError(
            "Este incremento solo admite origen sintético. Datos reales exigen otro protocolo."
        )
    if not 30 <= len(d) <= 5000:
        raise ValueError("Prototipo dimensionado para 30 a 5000 filas.")
    if set(y) != {0, 1, 2} or y.value_counts().min() < 10:
        raise ValueError("Faltan clases o casos por clase.")
    if d.comentario_atencion.str.casefold().duplicated().any():
        raise ValueError("Textos duplicados normalizados.")
    # Minimización: conservar solo los campos necesarios para reproducir el experimento.
    limpio = pd.DataFrame(
        {
            "id_atencion": d.id_atencion,
            "texto": d.comentario_atencion,
            "labels": y.astype(int),
            "origen_dato": d.origen_dato,
        }
    )
    # SHA-256 identifica los bytes del dataset; no es una métrica del modelo.
    calidad = {
        "filas": len(d),
        "columnas": list(d),
        "clases": d.etiqueta_sentimiento.value_counts().to_dict(),
        "nulos_obligatorios": 0,
        "duplicados": 0,
        "origen": "sintético",
        "etiquetado": "referencia sintética sin validación humana",
        "sha256_dataset": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return (limpio, calidad)


def agrupar_partir(d, semilla=42, umbral=0.8):
    """C2 · Crear grupos de textos similares y separar los datos sin cruzarlos.

    Hashing convierte fragmentos de caracteres en vectores sin aprender un
    vocabulario del corpus. No es el TF-IDF de la línea base.
    El coseno mide semejanza textual. Cada componente conexa forma un grupo:
    si A se parece a B y B a C, los tres permanecen en la misma partición.
    La estratificación intenta mantener proporciones de las clases.
    No equivale a ejecutar cinco entrenamientos de validación cruzada:
    usamos cinco bloques para obtener una sola división train/val/test.
    La matriz de similitud ocupa O(N²); por eso la carga limita el tamaño.
    """
    # Representación fija de fragmentos de 3 a 5 caracteres para detectar semejanzas.
    x = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
    ).transform(d.texto.str.casefold())
    # Con vectores normalizados L2, el producto escalar equivale al coseno.
    sim = (x @ x.T).toarray()
    # Convertir las semejanzas en un grafo y obtener grupos por conectividad.
    n, g = connected_components(csr_matrix(sim >= umbral), directed=False)
    for c in CLASES:
        if len(np.unique(g[d.labels.to_numpy() == c])) < 5:
            raise ValueError("Menos de cinco grupos por clase.")
    # Inicialmente ninguna fila pertenece a un bloque.
    fold = np.full(len(d), -1)
    for k, (_, idx) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=semilla).split(
            d.texto, d.labels, g
        )
    ):
        fold[idx] = k
    # Bloque 0 a test, bloque 1 a validación y los otros tres a entrenamiento.
    p = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))
    d = d.assign(grupo=g, particion=p)
    for nombre in ["train", "val", "test"]:
        if set(d.loc[d.particion == nombre, "labels"]) != {0, 1, 2}:
            raise ValueError("Falta clase en " + nombre)
    # Auditar que no quede un par sobre el umbral en particiones distintas.
    max_sim = float(sim[p[:, None] != p[None, :]].max())
    if max_sim >= umbral + 1e-08:
        raise RuntimeError("Fallo de separación de grupos.")
    return (
        d,
        {
            "grupos": int(n),
            "pares_similares": int(np.triu(sim >= umbral, 1).sum()),
            "max_coseno_entre_particiones": max_sim,
            "umbral": umbral,
            "semilla": semilla,
            "conteos": pd.crosstab(d.particion, d.labels).to_dict(),
            "limite": "La similitud de caracteres no identifica todas las familias sintéticas.",
        },
    )


def metricas(y, p):
    """C4 · Comparar etiquetas de referencia y predicciones.

    y: respuesta de referencia. p: clase predicha por el modelo.
    Accuracy: aciertos totales / número de casos.
    Precision por clase: aciertos de esa clase / predicciones de esa clase.
    Recall por clase: aciertos de esa clase / casos reales de esa clase.
    Macro-F1: promedio de las F1 de las tres clases con igual peso.
    Matriz: filas = referencia, columnas = predicción; diagonal = aciertos.
    """
    return {
        "accuracy": float(accuracy_score(y, p)),
        "f1_macro": float(
            f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "por_clase": classification_report(
            y,
            p,
            labels=[0, 1, 2],
            target_names=list(CLASES.values()),
            output_dict=True,
            zero_division=0,
        ),
        "matriz_confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
    }


def entrenar_baseline(train, val, out, semilla):
    """C3a · Entrenar TF-IDF + regresión logística y elegir C con validación.

    TF-IDF transforma palabras y pares de palabras en valores numéricos.
    La regresión logística aprende a relacionar esos valores con las etiquetas.
    C controla la regularización: un C menor supone mayor regularización.
    Se prueban tres candidatos; se guarda el de mayor macro-F1 de validación.
    El vocabulario y los pesos IDF se aprenden únicamente con entrenamiento.
    """
    import joblib

    best, score, hist = (None, -1, [])
    # Crear un pipeline nuevo por candidato para no arrastrar un ajuste anterior.
    for c in [0.5, 1.0, 2.0]:
        m = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=20000,
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=semilla,
                    ),
                ),
            ]
        )
        # Aprender SOLO con train: ni validación ni test ajustan este pipeline.
        m.fit(train.texto, train.labels)
        f = metricas(val.labels, m.predict(val.texto))["f1_macro"]
        hist.append({"C": c, "macro_f1_val": f})
        # Usar > conserva el primer candidato en un empate exacto.
        if f > score:
            best, score = (m, f)
    joblib.dump(best, out / "baseline.joblib")
    json_out(out / "seleccion_baseline.json", hist)
    return best


def entrenar_beto(train, val, out, a):
    """C3b · Ajustar BETO y devolver una función de predicción.

    El tokenizador convierte texto en IDs de tokens y máscaras de atención.
    BETO parte de pesos preentrenados en español; la cabeza de clasificación
    se adapta a tres clases. Se ajustan todos los pesos con nuestros datos train.
    Una época es una pasada completa por entrenamiento; un lote agrupa textos
    para calcular la pérdida y actualizar pesos.
    Al terminar se recarga la época elegida por validación, no por test.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError("Instale torch y transformers según cabecera.") from exc
    # Semillas y algoritmos deterministas reducen variación; GPU puede variar numéricamente.
    torch.manual_seed(a.semilla)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.semilla)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Elegir CPU o CUDA según la opción y la disponibilidad del equipo.
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if a.dispositivo == "auto"
        else a.dispositivo
    )
    if device == "cuda" and (not torch.cuda.is_available()):
        raise ValueError("CUDA no disponible.")
    # Si los pesos ya están en una carpeta local, no solicitar una revisión remota.
    rev = {} if Path(a.checkpoint).is_dir() else {"revision": a.revision}
    # Cargar tokenizador y pesos. La primera vez puede requerir descarga.
    tok = AutoTokenizer.from_pretrained(a.checkpoint, **rev)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.checkpoint, **rev, num_labels=3, id2label=CLASES, label2id=CODIGOS
    ).to(device)
    revision = getattr(model.config, "_commit_hash", None)
    # AdamW actualiza pesos usando los gradientes de la pérdida.
    opt = torch.optim.AdamW(model.parameters(), lr=2e-05, weight_decay=0.01)

    # Función interna: convierte una lista de textos en clases, sin entrenar.
    def predict(texts):
        # Modo evaluación: desactivar mecanismos de entrenamiento como dropout.
        model.eval()
        pred = []
        # La predicción no calcula gradientes ni modifica pesos.
        with torch.no_grad():
            for ini in range(0, len(texts), a.lote):
                b = tok(
                    list(texts[ini : ini + a.lote]),
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                # argmax elige la clase con mayor puntuación; no es un porcentaje de satisfacción.
                pred.extend(
                    model(**{k: v.to(device) for k, v in b.items()})
                    .logits.argmax(-1)
                    .cpu()
                    .tolist()
                )
        return np.asarray(pred)

    # Generador reproducible para barajar el orden de train en cada época.
    rng = np.random.default_rng(a.semilla)
    best = -1
    hist = []
    texts = train.texto.tolist()
    y = train.labels.to_numpy()
    for epoch in range(1, a.epocas + 1):
        # Reactivar el modo entrenamiento al inicio de cada época.
        model.train()
        order = rng.permutation(len(train))
        loss_sum = 0
        for ini in range(0, len(order), a.lote):
            ids = order[ini : ini + a.lote]
            b = tok(
                [texts[i] for i in ids],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            # Borrar gradientes anteriores antes de calcular los del lote actual.
            opt.zero_grad(set_to_none=True)
            # Pérdida supervisada: comparar salida del modelo con las etiquetas del lote.
            loss = model(
                **{k: v.to(device) for k, v in b.items()},
                labels=torch.tensor(y[ids], dtype=torch.long, device=device),
            ).loss
            # Propagar el error para calcular cómo debe cambiar cada peso.
            loss.backward()
            # Limitar la norma de los gradientes a 1 para estabilizar actualizaciones.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Aplicar una actualización a los pesos.
            opt.step()
            loss_sum += float(loss.detach().cpu()) * len(ids)
        # Medir validación al terminar la época; no usar test para decidir.
        f = metricas(val.labels, predict(val.texto.tolist()))["f1_macro"]
        hist.append(
            {"epoca": epoch, "loss_train": loss_sum / len(train), "macro_f1_val": f}
        )
        logging.info("BETO época %s: macro-F1 val %.4f", epoch, f)
        # Guardar únicamente una mejora estricta de macro-F1 de validación.
        if f > best:
            best = f
            model.save_pretrained(out / "beto")
            tok.save_pretrained(out / "beto")
    # Recargar el mejor checkpoint, que puede no ser el de la última época.
    model = AutoModelForSequenceClassification.from_pretrained(out / "beto").to(device)
    json_out(out / "seleccion_beto.json", hist)
    json_out(
        out / "config_beto.json",
        {
            "checkpoint": a.checkpoint,
            "revision": revision,
            "dispositivo": device,
            "max_length": 128,
            "lote": a.lote,
            "epocas": a.epocas,
            "lr": 2e-05,
            "weight_decay": 0.01,
        },
    )
    return predict


def comparar(y, pa, pb, g, semilla):
    """C4 · Comparar predicciones emparejadas de los mismos comentarios.

    El bootstrap remuestrea grupos completos con reemplazo y calcula un
    intervalo de la diferencia de macro-F1. Un intervalo que contiene cero
    no demuestra una mejora concluyente.
    McNemar contrasta errores discordantes; no es otra métrica de F1.
    Ambos análisis son exploratorios por el origen sintético del corpus.
    No es un tercer clasificador: aquí no se entrena ningún modelo adicional.
    """
    y, pa, pb, g = map(np.asarray, [y, pa, pb, g])
    rng = np.random.default_rng(semilla)
    groups = np.unique(g)
    indices = {v: np.flatnonzero(g == v) for v in groups}
    diff = []
    # Remuestreo pareado: ambos modelos reciben exactamente los mismos índices.
    for _ in range(1000):
        ix = np.concatenate(
            [indices[v] for v in rng.choice(groups, len(groups), replace=True)]
        )
        diff.append(
            metricas(y[ix], pb[ix])["f1_macro"] - metricas(y[ix], pa[ix])["f1_macro"]
        )
    # b: solo acierta baseline; c: solo acierta BETO. Son errores discordantes.
    b = int(((pa == y) & (pb != y)).sum())
    c = int(((pa != y) & (pb == y)).sum())
    return {
        "delta_f1": metricas(y, pb)["f1_macro"] - metricas(y, pa)["f1_macro"],
        "ic95_grupos": np.quantile(diff, [0.025, 0.975]).tolist(),
        "replicas": 1000,
        "mcnemar_p": float(binomtest(c, b + c, 0.5).pvalue) if b + c else 1.0,
        "b": b,
        "c": c,
        "limite": "Sintético. McNemar supone independencia entre casos; no garantizada. No prueba eficacia clínica.",
    }


def exportar(out, test, pred, summary):
    """C5 · Guardar métricas por clase, matrices, predicciones y reporte HTML.

    pred contiene solo los modelos realmente ejecutados: baseline y/o beto.
    Cada nombre produce sus propios JSON, CSV y PNG.
    La tabla de classification_report también incluye promedios y accuracy;
    son resúmenes de métricas, no modelos adicionales.
    """
    import matplotlib

    # Generar imágenes sin abrir ventanas; permite ejecución desde consola.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = test[["id_atencion", "texto", "labels", "grupo"]].copy()
    sections = []
    # Exportar una sección por modelo presente en el diccionario de predicciones.
    for name, p in pred.items():
        m = metricas(test.labels, p)
        summary[name] = m
        rows["pred_" + name] = p
        json_out(out / f"metricas_{name}.json", m)
        table = pd.DataFrame(m["por_clase"]).T
        table.to_csv(out / f"metricas_clase_{name}.csv", encoding="utf-8-sig")
        # Construir la matriz 3×3 con el mismo orden de etiquetas en ambos ejes.
        cm = np.asarray(m["matriz_confusion"])
        pd.DataFrame(cm, index=CLASES.values(), columns=CLASES.values()).to_csv(
            out / f"matriz_{name}.csv", encoding="utf-8-sig"
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=CLASES.values(),
            yticklabels=CLASES.values(),
            xlabel="Predicción",
            ylabel="Referencia sintética",
            title=name,
        )
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        fig.tight_layout()
        fig.savefig(out / f"matriz_{name}.png", dpi=170)
        plt.close(fig)
        sections.append(
            f"<h2>{name}</h2><p>Accuracy {m['accuracy']:.4f} · Macro-F1 {m['f1_macro']:.4f}</p>"
            + table.round(4).to_html()
            + f'<img width="480" src="matriz_{name}.png">'
        )
    # Guardar predicciones por registro para poder revisar los errores.
    rows.to_csv(out / "predicciones_test.csv", index=False, encoding="utf-8-sig")
    json_out(out / "resumen.json", summary)
    (out / "reporte.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><title>PI8 T6</title><style>body{font:16px Arial;max-width:1000px;margin:40px auto}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>PI8 · Análisis de sentimientos</h1><p>Elvis Max Estela Paucar · T6 · Grupo A</p><p>Corpus sintético de prueba. No valida pacientes reales. BETO: '
        + html.escape(summary["beto_estado"])
        + "</p>"
        + "".join(sections)
        + "<h2>Comparación</h2><pre>"
        + html.escape(
            json.dumps(summary.get("comparacion", {}), ensure_ascii=False, indent=2)
        )
        + "</pre><p>Las etiquetas no tienen validación humana. Puede persistir semejanza entre plantillas.</p></html>",
        encoding="utf-8",
    )


def autoprueba():
    """Verificar reglas de carga con un CSV temporal, sin entrenar modelos.

    Los datos pequeños de esta función son solo casos de prueba del programa.
    No se usan como entrenamiento ni sustituyen al Excel del proyecto.
    """
    import tempfile

    assert normalizar("  No\tme   atendieron ") == "No me atendieron"
    assert metricas([0, 1, 2], [0, 1, 2])["f1_macro"] == 1
    data = [
        {
            "id_atencion": f"T-{i}",
            "comentario_atencion": f"Texto único {i}",
            "etiqueta_sentimiento": CLASES[i % 3],
            "codigo_sentimiento": i % 3,
            "origen_dato": "sintético",
        }
        for i in range(30)
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.csv"
        pd.DataFrame(data).to_csv(p, index=False)
        assert len(cargar_validar(p)[0]) == 30
        for col, val in [
            ("codigo_sentimiento", 2),
            ("comentario_atencion", " "),
            ("id_atencion", "T-1"),
            ("origen_dato", "real"),
        ]:
            d = pd.DataFrame(data)
            d.loc[0, col] = val
            d.to_csv(p, index=False)
            try:
                cargar_validar(p)
            except ValueError:
                pass
            else:
                raise AssertionError("No se rechazó entrada inválida: " + col)
    print("Autoprueba OK: normalización, métricas y cuatro rechazos críticos.")


def main():
    """C5 · Punto de entrada que coordina la ejecución de principio a fin.

    Lee opciones de consola, crea una carpeta de corrida y llama a C1–C4.
    Los archivos de salida permiten explicar y reproducir el experimento.
    Devuelve 0 si completa la corrida y 1 si se produce una excepción.
    """
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--datos", type=Path, default=root / "Dataset_atencion_1000_revisado.xlsx"
    )
    p.add_argument("--salida", type=Path, default=root / "resultados_PI8")
    # Modo por defecto: comparación completa. --modelo baseline permite una prueba rápida.
    p.add_argument("--modelo", choices=["baseline", "ambos"], default="ambos")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--umbral", type=float, default=0.8)
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--predecir")
    p.add_argument("--autoprueba", action="store_true")
    a = p.parse_args()
    if a.autoprueba:
        autoprueba()
        return 0
    if not 0.5 <= a.umbral <= 1 or a.epocas < 1 or a.lote < 1:
        p.error("Umbral en [0.5,1]; épocas y lote positivos.")
    # Carpeta nueva por ejecución: conserva las corridas anteriores sin sobrescribirlas.
    out = a.salida / datetime.now(timezone.utc).strftime("corrida_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out / "ejecucion.log", encoding="utf-8"),
        ],
        force=True,
    )
    start = time.perf_counter()
    try:
        # Paso 1: cargar y validar el Excel; registrar su calidad.
        np.random.seed(a.semilla)
        d, q = cargar_validar(a.datos)
        json_out(out / "calidad.json", q)
        # Paso 2: agrupar, dividir y guardar la auditoría de separación.
        d, audit = agrupar_partir(d, a.semilla, a.umbral)
        json_out(out / "auditoria_particion.json", audit)
        d.to_csv(out / "corpus_preparado.csv", index=False, encoding="utf-8-sig")
        d[["id_atencion", "grupo", "particion", "labels"]].to_csv(
            out / "particiones.csv", index=False
        )
        # Separar las tres tablas; con el Excel entregado son 600, 200 y 200.
        train, val, test = [d[d.particion.eq(n)] for n in ["train", "val", "test"]]
        logging.info("Train=%d, val=%d, test=%d", len(train), len(val), len(test))
        # Paso 3: entrenar y seleccionar la línea base; predecir el test reservado.
        m = entrenar_baseline(train, val, out, a.semilla)
        # Solo se registran los dos modelos del parcial; no se crea otro clasificador.
        pred = {"baseline": m.predict(test.texto)}
        summary = {
            "naturaleza": "sintético",
            "n_test": len(test),
            "beto_estado": "no ejecutado",
        }
        exportar(out, test, pred, summary)
        # Paso 4: ajustar BETO y evaluarlo con los mismos textos de test.
        if a.modelo == "ambos":
            fn = entrenar_beto(train, val, out, a)
            pred["beto"] = fn(test.texto.tolist())
            summary["beto_estado"] = "ejecutado"
            # Comparar predicciones ya calculadas; esto no vuelve a entrenar los modelos.
            summary["comparacion"] = comparar(
                test.labels, pred["baseline"], pred["beto"], test.grupo, a.semilla
            )
        # Demostración opcional con un texto nuevo, separada de las métricas de test.
        if a.predecir:
            t = normalizar(a.predecir)
            if not t:
                raise ValueError("Texto vacío para demostración.")
            demo = {"texto": t, "baseline": CLASES[int(m.predict([t])[0])]}
            if "beto" in pred:
                demo["beto"] = CLASES[int(fn([t])[0])]
            json_out(out / "demo.json", demo)
        exportar(out, test, pred, summary)
        # Paso 5: registrar versiones y hashes para reproducibilidad.
        versions = {}
        for n in [
            "numpy",
            "pandas",
            "openpyxl",
            "scikit-learn",
            "scipy",
            "matplotlib",
            "joblib",
            "torch",
            "transformers",
        ]:
            try:
                versions[n] = importlib.metadata.version(n)
            except importlib.metadata.PackageNotFoundError:
                pass
        json_out(
            out / "manifest.json",
            {
                "estado": "completo",
                "parametros": vars(a),
                "python": sys.version,
                "plataforma": platform.platform(),
                "versiones": versions,
                "segundos": time.perf_counter() - start,
                "sha256_dataset": q["sha256_dataset"],
                "sha256_script": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
        )
        (out / "requirements_ejecucion.txt").write_text(
            "\n".join((f"{k}=={v}" for k, v in versions.items())), encoding="utf-8"
        )
        (out / "README.md").write_text(
            "# PI8 T6\n\nAbrir reporte.html. Corpus sintético; no valida pacientes reales.\n\nRepetir con el mismo Excel, script, versiones y parámetros del manifest. GPU puede introducir variaciones numéricas.\n\nDemo: mostrar Guía del Excel, ejecutar, explicar matriz y macro-F1, mostrar particiones y manifest. Preparar video de 2–3 minutos y publicar repositorio por separado. No se simulan commits.\n",
            encoding="utf-8",
        )
        print("Resultados:", out.resolve())
        for name in pred:
            print(name, summary[name]["accuracy"], summary[name]["f1_macro"])
        return 0
    # Dejar una evidencia clara del fallo y devolver un código de error.
    except Exception as exc:
        logging.exception("Corrida incompleta")
        json_out(
            out / "error.json",
            {"estado": "incompleto", "tipo": type(exc).__name__, "mensaje": str(exc)},
        )
        print("Error:", exc, "Evidencia parcial:", out, file=sys.stderr)
        return 1


# Ejecutar main solo cuando este archivo se inicia desde la consola.
if __name__ == "__main__":
    raise SystemExit(main())
