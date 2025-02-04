import mendeleev
import re
import numpy as np
import scipy as sp

def get_mass_abundance_isotope(element,rank=1,verbose=1):
    isotopes = {}
    for iso in element.isotopes:
        if iso.abundance is None:
            continue
        isotopes[iso.mass] = iso.abundance/100.
    
    if rank > len(isotopes):
        if verbose == 1:
            print('ERROR, Not this many isotopes')
        return None, None
    
    # Set all abundances of higher ranking isotopes to -1 before we compute the maximum
    for i in np.arange(rank-1):
        mass = max(isotopes, key=isotopes.get)
        isotopes[mass] = -1
    
    abundance = max(isotopes.values())
    mass = max(isotopes, key=isotopes.get)
    
    return mass, abundance


def get_isotopologue(chem_form,max_rank=2,cutoff_abundance=1.e-5):
    '''
    The chemical formula is to contain the elements with the sum of their occurance, e.g. not HCHO but either H2CO or H2C1O1
    max rank: The rank of most abundant isotopes taken into account
    cutoff_abundance: Isotopes with an abundance of less than 0.001% are not considered
    '''
    # Works up to max_rank 2
    composition = re.findall('[A-Z][a-z]*[0-9]*',chem_form)    
    n_groups = len(composition)
    if n_groups >= 5:
        print('ERROR: Too many elements in compound, isotopologue not supported')
        raise ValueError

    elements = []
    for group in composition:
        elements.append(mendeleev.element(re.search(r'\D+',group).group()).symbol)

    if len(np.unique(elements)) != len(elements):
        print('ERROR: Elements repeated in chem_form, please sum all elements.')
        raise ValueError

    isotopologue = {}
    group_isotopologue = {}
    for group in composition:
        element = mendeleev.element(re.search(r'\D+',group).group())
        index = re.search(r'\d+', group)
        if not index is None:
            index = int(index.group())
        else:
            index = 1
        
        iso_elem = {}
        for i in np.arange(max_rank):
            mass, abundance = get_mass_abundance_isotope(element,rank=i+1,verbose=0)
            if mass is None:
                continue
            iso_elem[i] = [mass, abundance]
        
        iso_group = {}
        if max_rank == 1 or len(iso_elem) == 1:
            iso_group[mass] = abundance
        
        elif max_rank == 2 or len(iso_elem) == 2:
            for i in np.arange(index+1):
                j = index - i
                mass = i*iso_elem[0][0] + j*iso_elem[1][0]
                abundance = sp.special.comb(index,i) * iso_elem[0][1]**i * iso_elem[1][1]**j
                
                if abundance < cutoff_abundance:
                    continue
                iso_group[mass] = abundance
                
        elif max_rank == 3 or len(iso_elem) == 3:
            for i in np.arange(index+1):
                for j in np.arange(index+1-i):
                    k = index - i - j
                    mass = i*iso_elem[0][0] + j*iso_elem[1][0] + k*iso_elem[2][0]
                    abundance = sp.special.comb(index,i) * sp.special.comb(index-i,j) * iso_elem[0][1]**i * iso_elem[1][1]**j * iso_elem[2][1]**k

                    if abundance < cutoff_abundance:
                        continue
                    iso_group[mass] = abundance                
                
        else:
            print('ERROR: max_rank not supported')
            raise ValueError
                    
        group_isotopologue[group] = iso_group

    if n_groups == 1:
        isotopologue = group_isotopologue[composition[0]]
        
    elif n_groups == 2:
        for m1, a1 in group_isotopologue[composition[0]].items():
            for m2, a2 in group_isotopologue[composition[1]].items():
                mass = m1+m2
                abundance = a1*a2
                if abundance < cutoff_abundance:
                    continue
                isotopologue[mass] = abundance
                
    elif n_groups == 3:
        for m1, a1 in group_isotopologue[composition[0]].items():
            for m2, a2 in group_isotopologue[composition[1]].items():
                for m3, a3 in group_isotopologue[composition[2]].items():
                    mass = m1+m2+m3
                    abundance = a1*a2*a3
                    if abundance < cutoff_abundance:
                        continue
                    isotopologue[mass] = abundance

    elif n_groups == 4:
        for m1, a1 in group_isotopologue[composition[0]].items():
            for m2, a2 in group_isotopologue[composition[1]].items():
                for m3, a3 in group_isotopologue[composition[2]].items():
                    for m4, a4 in group_isotopologue[composition[3]].items():
                        mass = m1+m2+m3+m4
                        abundance = a1*a2*a3*a4
                        if abundance < cutoff_abundance:
                            continue
                        isotopologue[mass] = abundance
    
    return isotopologue

def get_multipliers(chem_form, max_rank=2, cutoff_abundance=1.e-5,normalise='total'):
    '''
    normalise:
      - total: normalise to total abundace
      - max: normalise to most abundent isotope (similar to sisweb)
    '''
    isotopologue = get_isotopologue(chem_form,max_rank,cutoff_abundance)

    if normalise=='total':
        total_abundance = 0
        for k, v in isotopologue.items():
            total_abundance += v
            
        norm = total_abundance

    elif normalise == 'max':
        norm = max(isotopologue.values())

    else:
        print('Error: normalisation is not recognised')
        raise ValueError

    res_mass_spectrum = {}
    for i in np.arange(len(isotopologue)):
        abundance = max(isotopologue.values())
        mass = max(isotopologue, key=isotopologue.get)
            
        res_mass_spectrum[mass] = (norm/abundance).round(2)
        isotopologue[mass] = -1
        
    return res_mass_spectrum