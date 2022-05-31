from deepvoice3_pytorch.tfcompat.hparam import HParams

# NOTE: Если вы хотите получить полный контроль над архитектурой модели. пожалуйста, взгляните
# посмотрите на код и измените все, что захотите. Некоторые гиперпараметры жестко заданы.

# Гиперпараметры по умолчанию:
hparams = HParams(
    name="deepvoice3",

    # Язык:
    # [en, jp]
    frontend='en',

    # Замените слова на их произношение с фиксированной вероятностью.
    #  например, "hello" на "HH AH0 L OW1"
    #  [en, jp]
    #  en: Слово -> произношение с помощью CMUDict
    #  jp: Слово -> произношение usnig MeCab
    #  [0 ~ 1.0]: 0 означает, что замена не происходит.
    replace_pronunciation_prob=0.5,

    # Удобный конструктор моделей
    # [deepvoice3, deepvoice3_multispeaker, ньянко]
    # Определения можно найти по адресу deepvoice3_pytorch/builder.py
    # deepvoice3: DeepVoice3 https://arxiv.org/abs/1710.07654
    # deepvoice3_multispeaker: многоголосная версия DeepVoice3
    # nyanko: https://arxiv.org/abs/1710.08969
    builder="deepvoice3",

    # Должно быть настроено в зависимости от используемого набора данных и модели
    n_speakers=1,
    speaker_embed_dim=16,

    # Аудио:
    num_mels=80,
    fmin=125,
    fmax=7600,
    fft_size=1024,
    hop_size=256,
    sample_rate=22050,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    # следует ли изменять масштаб формы сигнала или нет.
    #  Пусть x - входная форма сигнала, масштабированная форма сигнала y задается с помощью:
    # y = x / np.abs(x).max() * rescaling_max
    rescaling=False,
    rescaling_max=0.999,
    # mel-спектрограмма нормализуется до [0, 1] для каждого высказывания, и может произойти отсечение,
    # зависит от min_level_db и ref_level_db, вызывая шум отсечения.
    # Если значение False, добавляется утверждение, чтобы гарантировать, что отсечения не произойдет.
    allow_clipping_in_normalization=True,

    # Модель:
    downsample_step=4,  # должно быть 4, когда builder="nyanko"
    outputs_per_step=1,  # должно быть 1, когда builder="nyanko"
    embedding_weight_std=0.1,
    speaker_embedding_weight_std=0.01,
    padding_idx=0,
    # Максимальное количество вводимых значений длины текста
    # попробуйте установить большее значение, если вы хотите ввести очень длинный текст
    max_positions=512,
    dropout=1 - 0.95,
    kernel_size=3,
    text_embed_dim=128,
    encoder_channels=256,
    decoder_channels=256,
    # Note: большие каналы конвертера требуют значительных вычислительных затрат
    converter_channels=256,
    query_position_rate=1.0,
    # может быть вычислен с помощью `compute_timestamp_ratio.py `.
    key_position_rate=1.385,  # 2.37 для jsut
    key_projection=False,
    value_projection=False,
    use_memory_mask=True,
    trainable_positional_encodings=False,
    freeze_embedding=False,
    # Если True, используйте внутреннее представление декодера для входных данных postnet,
    # в противном случае используйте mel-спектрограмму.
    use_decoder_state_for_postnet_input=True,

    # Загрузчик данных
    pin_memory=True,
    num_workers=1,  # Set it to 1 when in Windows (MemoryError, THAllocator.c 0x5)

    # Потеря
    masked_loss_weight=0.5,  # (1-w)*loss + w * masked_loss
    priority_freq=3000,  # эвристика: priotrize [0 ~ priotiry_freq] для линейных потерь
    priority_freq_weight=0.0,  # (1-w)*linear_loss + w*priority_linear_loss
    # https://arxiv.org/pdf/1710.08969.pdf
    # Добавление расхождения к потере стабилизирует обучение, 
    # особенно для очень глубоких (> 10 слоев) сетей. 
    # Двоичная потеря div, похоже, имеет примерно 10-кратный масштаб по сравнению с потерей L1, поэтому я выбираю 0.1.
    binary_divergence_weight=0.1,  #  установите 0 для отключения
    use_guided_attention=True,
    guided_attention_sigma=0.2,

    # Обучение:
    batch_size=16,
    adam_beta1=0.5,
    adam_beta2=0.9,
    adam_eps=1e-6,
    amsgrad=False,
    initial_learning_rate=5e-4,  # 0.001,
    lr_schedule="noam_learning_rate_decay",
    lr_schedule_kwargs={},
    nepochs=2000,
    weight_decay=0.0,
    clip_thresh=0.1,

    # Сохранить
    checkpoint_interval=10000,
    eval_interval=10000,
    save_optimizer_state=True,

    # Оценка:
    # это может быть список для нескольких уровней внимания,
    # например, [True, False, False, False, True]
    force_monotonic_attention=True,
    # Ограничение внимания для инкрементного декодирования
    window_ahead=3,
    # 0, как правило, предотвращает повторение слов, но иногда приводит к пропуску слов
    window_backward=1,
    power=1.4,  # Мощность для увеличения значений до уровня, предшествующего восстановлению фазы

    # GC:
    # Вероятность принудительной сборки мусора
    # Использовать только тогда, когда ошибка памяти продолжается в Windows (по умолчанию отключена)
    #gc_probability = 0.001,

    # только в режиме json_meta
    # 0: "использовать все",
    # 1: "игнорировать только несоответствующее выравнивание (unmatched_alignment)",
    # 2: "полностью игнорировать распознавание",
    ignore_recognition_level=2,
    # при работе с не выделенным речевым набором данных (например, отрывками из фильмов) желательно установить значение min_text выше 15. 
    # Может быть скорректирован по набору данных.
    min_text=20,
    # если значение true, данные без файла выравнивания фонем (.lab) будут проигнорированы
    process_only_htk_aligned=False,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
