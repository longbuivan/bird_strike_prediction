import pandas as pd
import numpy as np

def add_dummy(df, col, pref):
    one_hot = pd.get_dummies(df[col], prefix=pref)
    df_enriched = pd.concat([df, one_hot], axis=1)
    return df_enriched


def clean_data(df, with_label=True):
    df = df.set_index('Record_ID')

    df['nb_strikes'] = df['Engine1_Strike'] + df['Engine2_Strike'] + df['Engine3_Strike'] + df['Engine4_Strike'] \
                       + df['Windshield_Strike'] + df['Nose_Strike'] + df['Radome_Strike'] + df['Nose_Strike'] \
                       + df['Propeller_Strike'] + df['Wing_or_Rotor_Strike'] + df['Fuselage_Strike'] + df[
                           'Landing_Gear_Strike'] \
                       + df['Tail_Strike'] + df['Lights_Strike'] + df['Other_Strike']
    df = add_dummy(df, 'Species_ID', 'SP')
    df = add_dummy(df, 'Visibility', 'VI')
    df = add_dummy(df, 'Flight_Phase', 'FP')
    df = add_dummy(df, 'Engines', 'engines')
    df = add_dummy(df, 'Engine_Model', 'emodel')
    df = add_dummy(df, 'Engine_Make', 'emake')
    df = add_dummy(df, 'Aircraft_Type', 'type')
    df = add_dummy(df, 'Aircraft_Mass', 'mass')
    df = add_dummy(df, 'Operator_ID', 'OPERATOR')
    df = df.fillna(value={'Precipitation': 'NONE'})
    dummies = pd.get_dummies(df.Precipitation, prefix='prec')
    dummies['prec_RAIN'] = dummies['prec_FOG, RAIN'] + dummies['prec_RAIN, SNOW'] + dummies['prec_FOG, RAIN, SNOW']
    dummies['prec_FOG'] = dummies['prec_FOG, RAIN'] + dummies['prec_FOG, SNOW'] + dummies['prec_FOG, RAIN, SNOW']
    dummies['prec_SNOW'] = dummies['prec_RAIN, SNOW'] + dummies['prec_FOG, SNOW'] + dummies['prec_FOG, RAIN, SNOW']
    dummies = dummies[['prec_NONE', 'prec_RAIN', 'prec_FOG', 'prec_SNOW']]
    df = pd.concat([df, dummies], axis=1)
    df['Warning_Issued'] = df['Warning_Issued'].str.upper()
    df['Warning_Issued'] = df['Warning_Issued'].map({'Y': 1, 'N': 0})

    cols = [
        #     'Aircraft_Mass',
        'Engines',
        #         'Warning_Issued',
        #     'Height',

        'Speed',
        'Aircraft_Damage',
        #         'Radome_Damage',
        'Windshield_Damage', 'Nose_Damage',
        'Engine1_Damage', 'Engine2_Damage',
        #     'Engine3_Damage', 'Engine4_Damage',
        'Engine_Ingested', 'Propeller_Damage', 'Wing_or_Rotor_Damage',
        'Fuselage_Damage', 'Landing_Gear_Damage', 'Tail_Damage',
        #    'Lights_Damage',
        'Other_Damage',
        'prec_NONE',
        #         'prec_RAIN', 'prec_FOG','prec_SNOW',
        'VI_DAY',
        'VI_NIGHT',
        'FP_APPROACH',
        'FP_CLIMB',
        'FP_EN ROUTE',
        'FP_LANDING ROLL',
        'FP_TAKEOFF RUN',
        'OPERATOR_AAL',
        'OPERATOR_ABX',
        'OPERATOR_AWE',
        'OPERATOR_BUS',
        'OPERATOR_FDX',
        'OPERATOR_GOV',
        'OPERATOR_JBU',
        'OPERATOR_MIL',
        'OPERATOR_PVT',
        'OPERATOR_SWA',
        'OPERATOR_UPS',
        'OPERATOR_USCBP',
        'OPERATOR_USCG',
        #     'type_A',
        'emake_7.0',
        'emake_10.0',
        'emake_13.0',
        'emake_19.0',
        'emake_22.0',
        'emake_31.0',
        'emake_34.0',
        'emake_43.0',
        'emodel_19',
        #     'engines_1.0', 'engines_2.0',
        #     'AIRPORT_KMEM', 'AIRPORT_KSDF', 'AIRPORT_KSPS', 'AIRPORT_KYNG', 'AIRPORT_ZZZZ',
        'SP_1G11', 'SP_J2204', 'SP_K1002', 'SP_NE1', 'SP_UNKBL', 'SP_UNKBS', 'SP_YI005',
        'nb_strikes',
    ]
    if with_label:
        cols.append('Impact')
    df = df[cols]
    return df
