#test de algoritmo seleccionador de ema
import time
import numpy as np

start = time.time()

    # Algoritmo seleccionador de ema mas cercana dependiendo si hay sobrecompra o sobreventa 

close = 17163 
low = 17155.7
high = 17163.3
ema20_15m = 17148.5
ema100_15m = 17139.2
ema200_15m = 17107
ema20_1h = 17143.2
ema100_1h = 17060
ema200_1h = 16989.4
ema20_4h = 17081
ema100_4h = 16955.2
ema200_4h = 17296.1
enter_long = 0
enter_short = 1

emas = list([ema20_15m, ema100_15m, ema200_15m, ema20_1h, ema100_1h, ema200_1h, ema20_4h, ema100_4h, ema200_4h])
        
ema_mas_cercana = list([abs(close-ema20_15m), abs(close-ema100_15m), abs(close-ema200_15m), abs(close-ema20_1h), abs(close-ema100_1h), abs(close-ema200_1h), abs(close-ema20_4h), abs(close-ema100_4h), abs(close-ema200_4h)])

emas_str = list(['ema20_15m', 'ema100_15m', 'ema200_15m', 'ema20_1h', 'ema100_1h', 'ema200_1h', 'ema20_4h', 'ema100_4h', 'ema200_4h'])

while True:
    print('hola')
    indice_ema_mas_cercana = int(ema_mas_cercana.index(min(ema_mas_cercana)))
    if enter_long == 1 and low > emas[indice_ema_mas_cercana]:
        print('hola1')
        print(emas_str[indice_ema_mas_cercana], ' ', emas[indice_ema_mas_cercana])
        new_entryprice = emas[indice_ema_mas_cercana] 
        break
    elif enter_short == 1 and high < emas[indice_ema_mas_cercana]:
        print('hola2')
        print(emas_str[indice_ema_mas_cercana], ' ', emas[indice_ema_mas_cercana])
        new_entryprice = emas[indice_ema_mas_cercana] 
        break
    else:
        print('hola5')
        print(emas)
        print(ema_mas_cercana)
        print(emas_str)
        print(indice_ema_mas_cercana)
        emas.pop(indice_ema_mas_cercana)
        ema_mas_cercana.pop(indice_ema_mas_cercana)
        emas_str.pop(indice_ema_mas_cercana)

#guardo el anterior por las dudas (se arreglo usando los metodos de list en vez de numpy array)
        #while True:
        #    print('hola')
        #    indice_ema_mas_cercana = int(np.where(ema_mas_cercana == np.min(ema_mas_cercana))[0])
        #    if enter_long == 1 and low > emas[indice_ema_mas_cercana]:
        #        print('hola1')
        #        print(emas_str[indice_ema_mas_cercana], ' ', emas[indice_ema_mas_cercana])
        #        new_entryprice = emas[indice_ema_mas_cercana] 
        #        break
        #    elif enter_short == 1 and high < emas[indice_ema_mas_cercana]:
        #        print('hola2')
        #        print(emas_str[indice_ema_mas_cercana], ' ', emas[indice_ema_mas_cercana])
        #        new_entryprice = emas[indice_ema_mas_cercana] 
        #        break
        #    else:
        #        print('hola5')
        #        print(emas)
        #        np.delete(emas, indice_ema_mas_cercana)
        #        np.delete(ema_mas_cercana, indice_ema_mas_cercana)
        #        np.delete(emas_str, indice_ema_mas_cercana)

end = time.time()
print(start,' ',end)
