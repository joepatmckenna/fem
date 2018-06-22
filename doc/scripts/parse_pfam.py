import numpy as np
import pandas as pd
import Bio
from Bio import AlignIO
import os, urllib, gzip, re


def parse_pfam(data_dir):

    # pfam ftp faq: https://pfam.xfam.org/help#tabview=tab13
    pfam_current_release = 'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release'

    files = [
        'Pfam-A.full.gz',  # The full alignments of the curated families (~6GB)
        'pdbmap.gz',  # Mapping between PDB structures and Pfam domains. (~2MB)
    ]

    for f in files:
        local = os.path.join(data_dir, f)
        if not os.path.exists(local):
            remote = os.path.join(pfam_current_release, f)
            urllib.urlretrieve(remote, local)

    # protein families
    pfam_file = os.path.join(data_dir, 'pfam.npy')
    if os.path.exists(pfam_file):
        pfam = np.load(pfam_file)
    else:
        pfam = []
        with gzip.open(os.path.join(data_dir, 'Pfam-A.full.gz'), 'r') as f:
            for i, line in enumerate(f):
                if line[:7] == '#=GF AC':
                    pfam.append(line.split(' ')[4][:-1])
        pfam = np.array(pfam)
        np.save(pfam_file, pfam)

    len_pfam = len(pfam)

    pdb_map_file = os.path.join(data_dir, 'pdbmap.gz')
    names = ['pdb_id', 'chain', 'lig', 'name', 'pfam', 'pfam_protein_id', 'res']
    pdb_map = pd.read_csv(
        pdb_map_file,
        sep='\t',
        engine='python',
        header=None,
        names=names,
        dtype=str,
        compression='gzip')
    for name in names:
        pdb_map[name] = pdb_map[name].map(lambda x: x.rstrip(';'))
    pdb_map.set_index('pdb_id', inplace=True)

    pdb_map_pfam = pdb_map['pfam'].unique()

    alignments = AlignIO.parse(
        gzip.open(os.path.join(data_dir, 'Pfam-A.full.gz'), 'r'), 'stockholm')

    pfam_info_file = os.path.join(data_dir, 'pfam_info.npy')

    if os.path.exists(pfam_info_file):
        pfam_info = np.load(pfam_info_file)
        pfam_info = pd.DataFrame(
            data=pfam_info[:, 1:],
            index=pfam_info[:, 0],
            columns=['i', 'min_m', 'max_m', 'res', 'seq'])

    else:

        pfam_info = []

        for i, a in enumerate(alignments):

            pfam_dir = os.path.join(data_dir, 'Pfam-A.full',
                                    pfam[i].split('.')[0])

            if not pfam[i].split('.')[0] in pdb_map_pfam:
                continue

            if os.path.exists(pfam_dir):
                msa = np.load(os.path.join(pfam_dir, 'msa.npy'))
                m = np.array([len(np.unique(s)) for s in msa])
                pfam_info.append(
                    [pfam[i], i,
                     m.min(),
                     m.max(), msa.shape[0], msa.shape[1]])
                continue

            try:
                os.makedirs(pfam_dir)
            except:
                pass

            msa = np.array(a).T
            msa = np.array([[s.lower() for s in r] for r in msa])

            ids = np.array([re.split('[_/-]', ai.id) for ai in a])

            m = np.array([len(np.unique(s)) for s in msa])

            np.save(os.path.join(pfam_dir, 'msa.npy'), msa)
            np.save(os.path.join(pfam_dir, 'ids.npy'), ids)

            pfam_info.append(
                [pfam[i], i,
                 m.min(),
                 m.max(), msa.shape[0], msa.shape[1]])

        pfam_info = np.array(pfam_info)
        np.save(pfam_info_file, pfam_info)

    return pfam, pfam_info, pdb_map
