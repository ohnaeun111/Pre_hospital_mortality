import pandas as pd

def preprocess_trauma_data(df: pd.DataFrame) -> pd.DataFrame:
    record = pd.DataFrame(index=df.index)

    # 1. Timestamp parsing and latency calculation
    df['injury_time'] = pd.to_datetime(df['injury_timestamp'], errors='coerce')
    df['symptom_time'] = pd.to_datetime(df['symptom_timestamp'], errors='coerce')
    df['latency_seconds'] = (df['symptom_time'] - df['injury_time']).dt.total_seconds()
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
        if pd.isna(age):
            return 'unknown'
        return str(min(int(age // 5) + 1, 26))  # age group: 0–4 → 1, ..., 125+ → 26

    record['age_group'] = df['age_years'].apply(bin_age)

    # 3. Categorical one-hot encoding
    def one_hot_encode(record, col, mapping, prefix):
        for key, val in mapping.items():
            record[f"{prefix}_{key}"] = (df[col] == val).astype('uint8')

    # Gender
    record['gender_male'] = (df['gender'] == 'male').astype('uint8')

    # Example for one-hot maps
    intentionality_map = {
        'accident': 'unintentional',
        'suicide': 'intentional_self_harm',
        'assault': 'assault',
        'others': 'other',
        'unknown': 'unspecified'
    }
    one_hot_encode(record, 'intentionality', intentionality_map, 'intentionality')

    injury_type_map = {
        'blunt': 'blunt',
        'penetrating': 'penetrating',
        'burn': 'burn',
        'others': 'other',
        'unknown': 'unknown'
    }
    one_hot_encode(record, 'injury_type', injury_type_map, 'injury_type')

    # Repeat similarly for:
    # - injury_mechanism
    # - accident_location
    # - hospital_visit_route
    # - transport_mode
    # - insurance_type
    # - consciousness_level
    # - protective_equipment_use

    # 4. Vital sign categorization (example: SBP)
    def categorize_ranges(series, bins, labels, prefix):
        for i, label in enumerate(labels):
            lower = bins[i]
            upper = bins[i + 1]
            mask = (series >= lower) & (series < upper)
            record[f'{prefix}_{label}'] = mask.astype('uint8')

    # SBP: Systolic Blood Pressure
    categorize_ranges(
        df['systolic_bp'],
        bins=[-float('inf'), 0, 50, 76, 90, float('inf')],
        labels=['0', '1_49', '50_75', '76_89', '90+'],
        prefix='sbp'
    )

    # Additional vital signs: DBP, Pulse, Respiration, Temp, SpO2 (omitted here for brevity)

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
