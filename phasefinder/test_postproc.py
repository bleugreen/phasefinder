from val import test_postprocessing_f_measure


if __name__ == '__main__':
    data_path = '../stft_db_b_phase_cleaned.h5'
    f_measure, cmlt, amlt = test_postprocessing_f_measure(data_path)
    print(f"Postprocessing Results:")
    print(f"F-measure: {f_measure:.3f}")
    print(f"CMLt: {cmlt:.3f}")
    print(f"AMLt: {amlt:.3f}")