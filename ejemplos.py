from procesado_de_colores import histograma, dibujo_simple, \
    dibujar_triangulo, unir_colores, filtro_desuabizado, suma_y_resta_de_imagen, \
    dufuminar, aclarar_oscurecer_y_contrastar, transformaciones_de_colores
from procesado_de_formas import recortar_imagen_con_mascara, \
    transformaciones, zoom_imagen, extender_imagen, separar_fondo_de_cierto_tono_con_mascara, borrar_mancha_en_imagen, \
    quitado_de_fondo_inteligente, dibujar_cotornos, rotacion_sin_cortes, recortar_un_segmento, \
    simetrias, contornos_internos_y_externos, transformacion_de_perspectiva
from ejemplos_np import caracteristicas_del_array, actualizar_valores, leer_valores, matris_intermediara, \
    comparacion_de_matrices, creacion_de_matrices, crear_diagonal, algebra_de_arrays, rehacer_forma_y_extender_anadir, \
    tipos_de_datos_numericos, algebra, siclos_logicos, orden_de_array, comparar_arrays
from procesado_monocromatico import transformaciones_morfologicas, compuertas_logicas, monocromatismo_y_somras, \
    monocromatismo_edge, dilatacion_y_contracion_de_claro_y_oscuro, monocromatismo_erocion_vertical_horizontal, \
    quitar_sombra_binaria_inteligente_gauss
from reconocimiento_de_patrones import seleccion_de_formas_por_dilatacion_y_contraccion, \
    gradiente_de_bordes_por_colores, \
    quitar_ruido_suabizar_y_ver_bordes, rec_de_objetos_por_canny_y_contraste, \
    detectar_circulos, deteccion_de_vertices, deteccion_de_esquinas, \
    coincidencia_de_imagen_dentro_de_otra, encontrar_y_borrar_puntos_pequenos, \
    cordenadas_de_esquinas, rec_de_objetos_por_threshold_y_refinamineto, rec_de_objetos_por_color, \
    rec_de_objetos_por_orientaion_y_area, rec_de_objeto_mas_grande_y_cordenadas_de_exremos, \
    recorte_de_colores_con_mascara_muldtiple, contornos_conveccos, contornos_similares, \
    caracteristicas_de_los_contornos, encontrar_linea, figuras_demaciado_cercas, \
    detexion_de_defectos_de_cavidad_contador_de_dedos
from interrelaciones import distancias_relativas, centre_de_objeto, orden_carteciano_de_objetos, \
    deteccon_geometrica_de_objeto, encontrar_y_leer_7_segmentos, traking_objeto_en_video, cosas
from sistema_cv import video_play, leer_imagen, guardar_imagen, caprurar_video, caprurar_fotografia, click_en_imagen, \
    slider_tienpo_real, ejecutable_desde_consola_y_argumentos_al_script, ajustar_parametros_y_enviar_datos_mientras_corre_video


def ejemplos_numpy():
    caracteristicas_del_array()
    actualizar_valores()
    leer_valores()
    matris_intermediara()
    comparacion_de_matrices()
    creacion_de_matrices()
    crear_diagonal()
    algebra_de_arrays()
    rehacer_forma_y_extender_anadir()
    tipos_de_datos_numericos()
    algebra()
    siclos_logicos()
    orden_de_array()
    comparar_arrays()

def procesado_de_formas():
    recortar_un_segmento()
    recortar_imagen_con_mascara()
    dibujar_cotornos()
    contornos_internos_y_externos()
    transformaciones()
    rotacion_sin_cortes()
    transformacion_de_perspectiva()
    zoom_imagen()
    extender_imagen()
    separar_fondo_de_cierto_tono_con_mascara()
    quitado_de_fondo_inteligente()
    borrar_mancha_en_imagen()
    simetrias()

def procesado_monocromatico():
    compuertas_logicas()
    transformaciones_morfologicas()
    monocromatismo_y_somras()
    monocromatismo_edge()
    dilatacion_y_contracion_de_claro_y_oscuro()
    monocromatismo_erocion_vertical_horizontal()
    quitar_sombra_binaria_inteligente_gauss()

def procesado_de_colores():
    histograma()
    dibujo_simple()
    dibujar_triangulo()
    unir_colores()
    filtro_desuabizado()
    suma_y_resta_de_imagen()
    dufuminar()
    aclarar_oscurecer_y_contrastar()
    transformaciones_de_colores()

def sistema_cv():
    video_play()
    leer_imagen()
    guardar_imagen()
    caprurar_video()
    caprurar_fotografia()
    click_en_imagen()
    slider_tienpo_real()
    ajustar_parametros_y_enviar_datos_mientras_corre_video()
    ejecutable_desde_consola_y_argumentos_al_script()

def reconocimiento_de_patrones():
    seleccion_de_formas_por_dilatacion_y_contraccion()
    gradiente_de_bordes_por_colores()
    recorte_de_colores_con_mascara_muldtiple()
    quitar_ruido_suabizar_y_ver_bordes()
    encontrar_linea()
    detectar_circulos()
    deteccion_de_vertices()
    deteccion_de_esquinas()
    coincidencia_de_imagen_dentro_de_otra()
    encontrar_y_borrar_puntos_pequenos()
    cordenadas_de_esquinas()
    rec_de_objetos_por_threshold_y_refinamineto()
    rec_de_objetos_por_canny_y_contraste()
    rec_de_objetos_por_color()
    rec_de_objetos_por_orientaion_y_area()
    rec_de_objeto_mas_grande_y_cordenadas_de_exremos()
    contornos_conveccos()
    contornos_similares()
    caracteristicas_de_los_contornos()
    figuras_demaciado_cercas()
    detexion_de_defectos_de_cavidad_contador_de_dedos()

def interrelaciones():
    distancias_relativas()
    centre_de_objeto()
    orden_carteciano_de_objetos()
    deteccon_geometrica_de_objeto()
    encontrar_y_leer_7_segmentos()
    traking_objeto_en_video()
    cosas()



# realidad aumentada
#https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html
