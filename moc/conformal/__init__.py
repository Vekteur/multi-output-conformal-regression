from .conformalizers import (
    M_CP, HDR_CP, HDR_H, DR_CP, L_CP, L_H, PCP, HD_PCP, C_PCP, CP2_PCP_Linear, 
)

conformalizers = {
    'M-CP': M_CP,
    'DR-CP': DR_CP,
    'HDR-CP': HDR_CP,
    'PCP': PCP,
    'HD-PCP': HD_PCP,
    'C-PCP': C_PCP,
    'CP2-PCP-Linear': CP2_PCP_Linear,
    'L-CP': L_CP,
    'HDR-H': HDR_H,
    'L-H': L_H,
}
