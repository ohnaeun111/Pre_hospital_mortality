import pandas as pd

def preprocess_trauma_data(df: pd.DataFrame) -> pd.DataFrame:
    record = pd.DataFrame(index=df.index)

    # 1. Timestamp parsing and latency calculation
    df['injury_time'] = pd.to_datetime(df['injury_timestamp'], errors='coerce')
    df['injury_hour'] = df['injury_time'].dt.hour

    def categorize_time(hour):
        if pd.isna(hour):
            return 'missing'
        return 'day' if 9 <= hour < 17 else 'night'

    time_category = df['injury_hour'].apply(categorize_time)
    for cat in ['day', 'night', 'missing']:
        record[f'injury_time_{cat}'] = (time_category == cat).astype('uint8')

    # 2. Age binning
    def bin_age(age):
        return str(min(int(age // 5) + 1, 26))  # age group: 0–4 → 1, ..., 125+ → 26

    record['age_group'] = df['age_years'].apply(bin_age)

    # 3. Categorical one-hot encoding
    def one_hot_encode(record, col, mapping, prefix):
        for key, val in mapping.items():
            if val == 'NaN':
                record[f"{prefix}_{key}"] = df[col].isna().astype('uint8')
            else:
                record[f"{prefix}_{key}"] = (df[col] == val).astype('uint8')

    # Gender
    record['gender_male'] = (df['gender'] == 'male').astype('uint8')

    # Example for one-hot maps
    intentionality_map = {
        'intentionality_accident': 'unintentional',
        'intentionality_suicide': 'intentional_self_harm',
        'intentionality_assault': 'assault',
        'intentionality_others': 'other',
        'intentionality_unknown': 'unspecified',
        'intentionality_missing': 'NaN'
    }
    one_hot_encode(record, 'intentionality', intentionality_map, 'intentionality')

    injury_type_map = {
        'injury_blunt': 'blunt',
        'injury_penetrating': 'penetrating',
        'injury_fire': 'burn',
        'injury_others': 'other',
        'injury_unknown': 'unknown',
        'injury_missing_data': 'NaN'
    }
    one_hot_encode(record, 'injury_type', injury_type_map, 'injury_type')

    # Repeat similarly for:
    # - injury_mechanism
    # - job related
    # - accident_location
    # - hospital_visit_route
    # - transport_mode
    # - insurance_type
    # - prearrival cardiac arrest
    # - prearrival Report
    # - response 
    # - protective_equipment_use
    

    # 4. Vital sign categorization (SBP, DBP, Pulse, Resp, Temp, SpO2)
    # Note: All NaN values in vital signs are assumed to be already replaced with the mean from the train dataset.
    #       Therefore, we can directly proceed with range-based categorization.
    def categorize_ranges(series, bins, labels, prefix):
        for i, label in enumerate(labels):
            lower = bins[i]
            upper = bins[i + 1]
            mask = (series >= lower) & (series < upper)
            record[f'{prefix}_{label}'] = mask.astype('uint8')

    # SBP
    categorize_ranges(
        df['systolic_bp'],
        bins=[-float('inf'), 0, 50, 76, 90, float('inf')],
        labels=['0', '1_49', '50_75', '76_89', '90_plus'],
        prefix='sbp'
    )

    # DBP
    categorize_ranges(
        df['diastolic_bp'],
        bins=[-float('inf'), 0, 30, 46, 60, float('inf')],
        labels=['0', '1_29', '30_45', '46_59', '60_plus'],
        prefix='dbp'
    )

    # Pulse
    categorize_ranges(
        df['pulse'],
        bins=[-float('inf'), 0, 30, 60, 101, 120, float('inf')],
        labels=['0', '1_29', '30_59', '60_100', '101_119', '120_plus'],
        prefix='pulse'
    )

    # Respiration
    categorize_ranges(
        df['respiration'],
        bins=[-float('inf'), 0, 6, 10, 30, float('inf')],
        labels=['0', '1_5', '6_9', '10_29', '30_plus'],
        prefix='resp'
    )

    # Body Temperature (°C)
    categorize_ranges(
        df['body_temp'],
        bins=[-float('inf'), 0, 24.0, 28.0, 32.0, 35.0, 37.8, float('inf')],
        labels=['0', '0_24', '24_28', '28_32', '32_35', '35_37', '38_plus'],
        prefix='temp'
    )

    # SpO2 (%)
    categorize_ranges(
        df['spo2'],
        bins=[-float('inf'), 0, 80, 91, 96, float('inf')],
        labels=['0', '1_80', '81_90', '91_95', '96_plus'],
        prefix='spo2'
    )

    # -1, -9
    def make_flags(col, name_prefix):
        record[f'{name_prefix}_uncheckable'] = (df[col] == -1).astype('uint8')
        record[f'{name_prefix}_unchecked'] = (df[col] == -9).astype('uint8')

    make_flags('systolic_bp', 'sbp')
    make_flags('diastolic_bp', 'dbp')
    make_flags('pulse', 'pulse')
    make_flags('respiration', 'resp')
    make_flags('body_temp', 'temp')
    make_flags('spo2', 'spo2')
    return record
