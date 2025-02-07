from .conformalizers import (
    M_CP, C_HDR, HDR_H, DR_CP, L_CP, L_H, PCP, HD_PCP, C_PCP, CP2_PCP_Linear, STDQR, CopulaCPTS
)

conformalizers = {
    'M-CP': M_CP,
    'DR-CP': DR_CP,
    'C-HDR': C_HDR,
    'PCP': PCP,
    'HD-PCP': HD_PCP,
    'C-PCP': C_PCP,
    'CP2-PCP-Linear': CP2_PCP_Linear,
    'L-CP': L_CP,
    'HDR-H': HDR_H,
    'L-H': L_H,
    'STDQR': STDQR,
    'CopulaCPTS': CopulaCPTS,
}
