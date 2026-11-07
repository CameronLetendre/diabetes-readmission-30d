"""
Semantic groupings for the three integer ID cols

raw codes are too fine-grained and many appear in
fewer than 50 rows, so their readmission rates are noise. Instead of 
Rather lumping every rare code into one "Other" bucket, each code is 
placed in a larger group whose members share the same clinical meaning, 
so the group keeps a trend.
"""

# admission_type_id: how urgently the patient was admitted
ADMISSION_TYPE = {
    1: "Emergency",      # Emergency
    7: "Emergency",      # Trauma Center (an emergency admission by definition)
    2: "Urgent",         # Urgent
    3: "Elective",       # Elective
    4: "Unknown",        # Newborn: ages in the data are 40-90, so a coding error
    5: "Unknown",        # Not Available
    6: "Unknown",        # NULL
    8: "Unknown",        # Not Mapped
}

# admission_source_id: where the patient came from
ADMISSION_SOURCE = {
    # Sent in by an outpatient provider (a planned, non-acute route in)
    1: "Referral",          # Physician Referral
    2: "Referral",          # Clinic Referral
    3: "Referral",          # HMO Referral

    # Walked in / brought in acutely
    7: "Emergency Room",    # Emergency Room

    # Already receiving institutional care somewhere else
    4: "Transfer",          # Transfer from a hospital
    5: "Transfer",          # Transfer from a Skilled Nursing Facility
    6: "Transfer",          # Transfer from another health care facility
    10: "Transfer",         # Transfer from critical access hospital
    18: "Transfer",         # Transfer from another Home Health Agency
    19: "Transfer",         # Readmission to same Home Health Agency
    22: "Transfer",         # Transfer from hospital inpt / same facility
    25: "Transfer",         # Transfer from Ambulatory Surgery Center
    26: "Transfer",         # Transfer from Hospice

    # No usable information
    8: "Unknown",           # Court / Law Enforcement (16 rows, no clinical meaning)
    9: "Unknown",           # Not Available
    15: "Unknown",          # Not Available
    17: "Unknown",          # NULL
    20: "Unknown",          # Not Mapped
    21: "Unknown",          # Unknown / Invalid
    11: "Unknown",          # Normal Delivery      } birth codes: impossible
    12: "Unknown",          # Premature Delivery   } for this cohort, 5 rows
    13: "Unknown",          # Sick Baby            } total, treated as
    14: "Unknown",          # Extramural Birth     } coding errors
    23: "Unknown",          # Born inside this hospital
    24: "Unknown",          # Born outside this hospital
}

# discharge_disposition_id: level of care the patient went to next.
DISCHARGE_DISPOSITION = {
    # Home, no formal support
    1: "Home",                      # Discharged to home

    # Home, but with clinical support coming to the house
    6: "Home with care",            # Home with home health service
    8: "Home with care",            # Home under care of Home IV provider

    # Long-term / custodial nursing care
    3: "Nursing facility",          # SNF (skilled nursing facility)
    4: "Nursing facility",          # ICF (intermediate care facility)
    24: "Nursing facility",         # Nursing facility, Medicaid-certified only

    # Sent to another acute-care hospital (needed a higher level of care)
    2: "Hospital",                  # Another short term hospital
    9: "Hospital",                  # Admitted as inpatient to this hospital
    10: "Hospital",                 # Neonate to another hospital (coding error, 6 rows)
    27: "Hospital",                 # Federal health care facility
    29: "Hospital",                 # Critical Access Hospital

    # Post-acute / specialty inpatient care that is not a nursing home
    5: "Rehab/LTC/Psych",           # Another type of inpatient care institution
    15: "Rehab/LTC/Psych",          # Medicare-approved swing bed
    22: "Rehab/LTC/Psych",          # Rehab facility
    23: "Rehab/LTC/Psych",          # Long term care hospital
    28: "Rehab/LTC/Psych",          # Psychiatric hospital / unit
    30: "Rehab/LTC/Psych",          # Health care institution not defined elsewhere

    # Patient chose to leave against medical advice (non-adherence signal)
    7: "Left AMA",

    # No usable information
    12: "Unknown",                  # Still patient / expected to return (3 rows)
    16: "Unknown",                  # Referred elsewhere for outpatient services (11 rows)
    17: "Unknown",                  # Referred here for outpatient services (14 rows)
    18: "Unknown",                  # NULL
    25: "Unknown",                  # Not Mapped
    26: "Unknown",                  # Unknown / Invalid

    # Cannot be readmitted. Rows with these codes are dropped in preprocessing.
    11: "Expired",
    19: "Expired",
    20: "Expired",
    21: "Expired",
    13: "Hospice",
    14: "Hospice",
}

ID_GROUPS = {
    "admission_type_id": ("admission_type", ADMISSION_TYPE),
    "admission_source_id": ("admission_source", ADMISSION_SOURCE),
    "discharge_disposition_id": ("discharge_disposition", DISCHARGE_DISPOSITION),
}


def apply_id_groups(df, drop_ids=True):
    df = df.copy()

    for id_col, (new_col, mapping) in ID_GROUPS.items():
        unmapped = set(df[id_col].unique()) - set(mapping)
        df[new_col] = df[id_col].map(mapping)

    if drop_ids:
        df = df.drop(columns=list(ID_GROUPS))
    return df
