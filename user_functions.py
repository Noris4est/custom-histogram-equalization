import numpy as np
import cv2 as cv


def calc_histogram(img):
    if len(img.shape) > 2:
        img = np.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img.flatten()  # reshape original array to 1D array
    bin_locations, counts = np.unique(img, return_counts=True)
    bin_locations.astype('int',copy=False)
    return bin_locations, counts


def calc_pdf_and_cdf_8bit(img):
    # Возвращает нормированные в диапазоне [0, 1] PDF и CDF распределения интенсивностей (для 8 битного изображения)
    bin_locations, counts = calc_histogram(img)
    img_pdf = counts/np.max(counts)
    full_pdf = np.zeros(256)
    for i in range(len(bin_locations)):
        full_pdf[bin_locations[i]] = img_pdf[i]
    full_cdf = np.cumsum(full_pdf)
    full_cdf = full_cdf/np.max(full_cdf)

    return np.arange(256), full_pdf, full_cdf


def image_equalize_classical(img_in, origin_colorspace = 'BGR'):
    """
    Реализован классический алгоритм эквализации гистограммы изображения.
    Входное изображение должно представлять собой numpy массив.
    Если входное изображение 3х канальное:
    1) оно обрабатывается как изображение в цветовом пространстве origin_colorspace.
    2) обрабатывается только яркостная (V) составляющая изображения,
    представленного в пространстве HSV.
    3) итоговое изображение возвращается в его исходном формате
    4) если функция не поддерживает переданное origin_colorspace,
    то будет возвращен код ошибки.
    :param img_in: одноканальная или 3х канальная матрица numpy
    :return: эквализованное изображение
    """
    if type(img_in) != np.ndarray:
        return 0
    if len(img_in.shape) > 2:
        # трехканальное изображение
        if origin_colorspace == 'BGR':
            img_hsv = cv.cvtColor(img_in, cv.COLOR_BGR2HSV)
            img_hsv[:, :, -1] = image_equalize_classical_single_ch(img_hsv[:, :, -1])
            img_out = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
        else:
            return 2
    else:
        # одноканальное изображение
        img_out = np.copy(img_in)
        img_out = image_equalize_classical_single_ch(img_out)
    return img_out


def image_equalize_classical_single_ch(img_in):
    if img_in.dtype == np.uint8:
        max_intensity = 2 ** 8 - 1
    elif img_in.dtype == np.uint16:
        max_intensity = 2 ** 16 - 1
    else:
        return 3
    bin_locations, counts = np.unique(img_in.flatten(), return_counts=True)
    cdf = np.cumsum(counts)
    cdf = cdf / np.max(cdf) * max_intensity
    img_out = np.interp(img_in, bin_locations, cdf).astype('uint8')
    return img_out


def img_equalize_flat_histogram_method1(img_in, num_bins=10000, origin_colorspace='BGR'):
    """
    The function equalizes the image histogram.
    The function accepts a single-channel or 3-channel image.
    In a 3-channel image, only the luminance channel is processed.
    The input image can have a coding depth of 8 or 16 bits. (uint8 or uint16).
    Increasing "num_bins" increases the amount of memory used and the total computation time.

    Equalization accuracy is a measure of the flatness of the histogram of the output image. Use num_bins > 256.
    if the number of channels is 3, then the luminance channel is equalized,
    and then image returned in its original format.

    Функция осуществляет эквализацию гистограммы изображения.
    На вход подается одноканальное или 3х канальное изображение.
    В 3х канальном изображении обрабатывается только канал яркости.
    Входное изображение может иметь глубину кодирования 8, 16 или 32 бита : uint8, uint16 или uint32.
    При увеличении "num_bins" увеличивается объем используемой памяти и общее время вычисления.

    :param img_in: input image - numpy array. Color channels 1 or 3.
    :param num_bins: equalization accuracy depends on the num_bins.
    :param origin_colorspace: if the number of image channels is 3, then it is processed in the origin_colorspace format.
    :return: returns an image with an equalized histogram.
    """
    if type(img_in) != np.ndarray:
        return 0
    if num_bins < 2:
        return 1
    if len(img_in.shape) > 2:
        # трехканальное изображение
        if origin_colorspace == 'BGR':
            img_hsv = cv.cvtColor(img_in, cv.COLOR_BGR2HSV)
            img_hsv[:, :, -1] = img_equalize_flat_histogram_method1_single_ch(img_hsv[:, :, -1], num_bins)
            img_out = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
        else:
            return 2
    else:
        # одноканальное изображение
        img_out = np.copy(img_in)
        img_out = img_equalize_flat_histogram_method1_single_ch(img_out, num_bins)
    return img_out


def img_equalize_flat_histogram_method1_single_ch(img_in, num_bins=10000):
    """
    Эквализация гистограммы одноканального изображения.
    Особенность реализации заключается в наложении реализации случайного шума с динамическим диапазоном [0, 1)
    Данное воздействие является обратимым, поэтому не считается искажающим. original_image = np.floor(noised_image).
    Шум позволяет бороться с проблемой квантования,
    таким образом у CDF кривой в большинстве случаев не будет плоских участков.
    (Из-за плоских областей CDF классический алгоритм для 8 битных изображений не выравнивает гистограмму полностью)
    :param img_in: numpy.ndarray (numpy массив)
    :param num_bins: число секторов разбиения динамического диапазона яркости пикселов.
    :return: изображение с эквализованной гистограммой, тип numpy.ndarray
    """
    if img_in.dtype == np.uint8:
        max_intensity = 2 ** 8 - 1
    elif img_in.dtype == np.uint16:
        max_intensity = 2 ** 16 - 1
    else:
        return 3
    img_n = img_in.astype(np.double) + np.random.rand(*list(img_in.shape))  # add random noise in range [0, 1)
    # this operation is reversible, because np.floor(im_n) = im_orig
    # quantization occurs when the signal is formed by digital camera

    bins_arr = np.linspace(0, max_intensity+1, num_bins)
    counts_im_n, edges_bin_loc_im_n = np.histogram(img_n.flatten(), bins=bins_arr)
    # bins = [0, ... , 256] => intervals [0, 1), [1, 2), [2, 3) ... [254, 255), [255, 256]
    bin_loc_img_n = edges_bin_loc_im_n[1:]

    cdf = np.cumsum(counts_im_n)
    cdf = cdf / np.max(cdf) * (max_intensity+1)

    img_eq = np.interp(img_n, bin_loc_img_n, cdf)
    img_eq = np.floor(img_eq)
    img_eq[img_eq > max_intensity] = max_intensity
    img_eq = img_eq.astype(dtype=np.uint8)
    return img_eq


def img_equalize_flat_histogram_method2(img_in, precision=2, origin_colorspace='BGR'):
    """
    The function equalizes the image histogram.
    The function accepts a single-channel or 3-channel image as input.
    In a 3-channel image, only the luminance channel is processed.
    The input image can have a coding depth of 8 or 16 bits. (uint8 or uint16).
    Increasing "precision" increases the amount of memory used and the total computation time.

    Функция осуществляет эквализацию гистограммы изображения.
    На вход подается одноканальное или 3х канальное изображение.
    В 3х канальном изображении обрабатывается только канал яркости.
    Входное изображение может иметь глубину кодирования 8, 16 или 32 бита : uint8, uint16 или uint32.
    При увеличении "precision" увеличивается объем используемой памяти и общее время вычисления.

    :param img_in: input image - numpy array. Color channels 1 or 3.
    :param precision: equalization accuracy depends on the precision. Equalization accuracy is a measure of the flatness of the histogram of the output image. Use precision = 0 ... 4
    :param origin_colorspace: if the number of image channels is 3, then it is processed in the origin_colorspace format
    :return: returns an image with an equalized histogram. if the number of channels is 3, then the luminance channel is equalized, and then image returned in its original format.
    """
    if type(img_in) != np.ndarray:
        return 0
    if precision < 0:
        return 1
    if len(img_in.shape) > 2:
        if origin_colorspace == 'BGR':
            img_hsv = cv.cvtColor(img_in, cv.COLOR_BGR2HSV)
            img_hsv[:, :, -1] = img_equalize_flat_histogram_method2_single_ch(img_hsv[:, :, -1], precision)
            img_out = cv.cvtColor(img_hsv,cv.COLOR_HSV2BGR)
        else:
            return 2
    else:
        # одноканальное изображение
        img_out = np.copy(img_in)
        img_out = img_equalize_flat_histogram_method2_single_ch(img_out, precision)
    return img_out


def img_equalize_flat_histogram_method2_single_ch(img_in, precision):
    if img_in.dtype == np.uint8:
        max_intensity = 2**8 -1
    elif img_in.dtype == np.uint16:
        max_intensity = 2**16 - 1
    else:
        return 3

    img_h, img_w = img_in.shape[:2]

    rand_vals_field = np.random.rand(img_h, img_w)  # матрица случайных значений [0,1)
    trunc_vals_field = np.round(rand_vals_field, precision)  # округление до требуемого знака после запятой
    img_n = img_in.astype(np.double) + trunc_vals_field  # noised image
    # this operation is reversible, because np.floor(im_n) = im_orig
    # quantization occurs when the signal is formed by digital camera
    '''
    Аналог расчета гистограммы, но вычисляется не количество значений выборки,
    попавших в установленные диапазоны. Для каждого уникального значения выборки (bin_locations)
    вычисляется количество вхождений (counts)
    '''
    bin_locations, counts = np.unique(img_n.flatten(), return_counts=True)

    '''
    Известно, что эквализация является градационным преобразованием над каждым пикселом в отдельности
    в зависимости от его яркости. Корреляция между соседними пикселами не учитывается.
    Градационное преобразование задается функцией преобразования.
    (см. Гонсалес Р., Вудс Р. Цифровая обработка изображений).
    Функция преобразования для эквализации является нормированной кумулятивной
    функцией распределения (CDF). counts является ненормированной PDF. 
    Для перехода из PDF в CDF осуществляется дискретное интегрирование.
    '''
    cs_counts = np.cumsum(counts)  # расчет ненормированной CDF
    cs_counts = cs_counts / np.max(cs_counts) * (max_intensity + 1)  # нормировка CDF
    hist_dict = dict(zip(list(bin_locations), list(cs_counts)))  # формирование функции преобразования
    img_eq = np.zeros(img_in.shape)  # формирование заготовки
    for i in range(img_h):
        for j in range(img_w):
            img_eq[i, j] = hist_dict[img_n[i, j]]
    img_eq = np.floor(img_eq)
    img_eq[img_eq > max_intensity] = max_intensity
    img_eq = img_eq.astype(dtype=np.uint8)
    return img_eq
