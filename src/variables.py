from config import root_path



config_params = {"utils_file_path": f'{root_path}/src',
"FinBERT_Tone_Model": 'FinBERT-FinVocab-Uncased',
"FinBERT_Tone_Model_Path": f'{root_path}/models/FinBERT-tone',
"trained_classification_layer": "model.pth",
"classification_layer_path": f"{root_path}/models/CNN_DNN_classifier/",
"weight_data_path": f'{root_path}data/news_weights.xlsx',
"company_sheet": 'nifty50_companies',
"sector_sheet": 'nifty50_industries',
"positive_word_sheet": 'unique_positve_words',
"negative_word_sheet": 'unique_negative_words',
"pos_bigram_sheet": 'unique_pos_bigram',
"neg_bigram_sheet": 'unique_neg_bigram',
"keyword_sheet": 'news_keywords',
"filtering_keywords": 'Keywords',
"stopwords_to_ignore": [],
"min": 0.1,
"max": 0.4}



model_parameters = {"input_channels": 1,
"output_channels": 16,
"conv_kernal_size": 3,
"pooling_kernal_size": 2,
"dnn_input_size": 16,
"dnn_output_size": 64,
"input_size": 768,
"num_classes": 3,
"max_words_in_three_sentence": 80,
"padding": 'max_length',
"class_mapping_of_sentiment": {2: -1, 0: 0, 1: 1},
"news_class_weight_matrix": [1,1,1,0.5,0.75],
"minimum_weight_to_assign": 0.4}



threshold_params = {"avg_past_max": 0.45,
"avg_past_mean": 0.27,
"avg_past_min": 0.08,
"avg_past_q3": 0.31,
"avg_past_std": 0.05,
"mean_all_news": 0.27,
"mean_minus_sigma_neg_news": -0.39,
"mean_neg_news": -0.22,
"mean_plus_sigma_all_news": 0.52,
"mean_plus_sigma_pos_news": 0.5,
"mean_pos_news": 0.34,
"q2_neg_news": -0.19,
"q3_all_news": 0.4,
"q3_neg_news": -0.06,
"q3_pos_news": 0.41,
"std_all_news": 0.24,
"std_neg_news": 0.17,
"std_pos_news": 0.16}

