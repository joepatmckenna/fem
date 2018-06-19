import numpy as np
import pandas as pd
import Bio
from Bio import AlignIO
import os, urllib, gzip


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
    pf_file = os.path.join(data_dir, 'pf.npy')
    if os.path.exists(pf_file):
        pf = np.load(pf_file)
    else:
        pf = []
        with gzip.open(os.path.join(data_dir, 'Pfam-A.full.gz'), 'r') as f:
            for i, line in enumerate(f):
                if line[:7] == '#=GF AC':
                    pf.append(line.split(' ')[4][:-1])
        pf = np.array(pf)
        np.save(pf_file, pf)

    pdbmap_file = os.path.join(data_dir, 'pdbmap.gz')
    names = ['pdbid', 'chain', '?', 'name', 'pf', 'id', 'res']
    pdbmap = pd.read_csv(
        pdbmap_file,
        sep='\t',
        engine='python',
        header=None,
        names=names,
        dtype=str,
        compression='gzip')
    for name in names:
        pdbmap[name] = pdbmap[name].map(lambda x: x.rstrip(';'))
    pdbmap.set_index('pdbid', inplace=True)

    pdbmap_pf = pdbmap['pf'].unique()

    alignments = AlignIO.parse(
        gzip.open(os.path.join(data_dir, 'Pfam-A.full.gz'), 'r'), 'stockholm')

    pf_info_file = os.path.join(data_dir, 'pf_info.npy')

    if os.path.exists(pf_info_file):
        pf_info = np.load(pf_info_file)
        pf_info = pd.DataFrame(
            data=pf_info[:, 1:],
            index=pf_info[:, 0],
            columns=['i', 'min_m', 'max_m', 'res', 'seq'])

    else:

        pf_info = []

        for i, a in enumerate(alignments):

            pf_dir = os.path.join(data_dir, 'Pfam', pf[i])

            if not pf[i].split('.')[0] in pdbmap_pf:
                continue

            if os.path.exists(pf_dir):
                msa = np.load(os.path.join(pf_dir, 'msa.npy'))
                m = np.array([len(np.unique(s)) for s in msa])
                pf_info.append(
                    [pf[i], i,
                     m.min(),
                     m.max(), msa.shape[0], msa.shape[1]])
                continue

            try:
                os.makedirs(pf_dir)
            except:
                pass

            msa = np.array(a).T
            msa = np.array([[s.lower() for s in r] for r in msa])

            ids = np.array([ai.id for ai in a])

            m = np.array([len(np.unique(s)) for s in msa])

            np.save(os.path.join(pf_dir, 'msa.npy'), msa)
            np.save(os.path.join(pf_dir, 'ids.npy'), ids)

            pf_info.append(
                [pf[i], i,
                 m.min(),
                 m.max(), msa.shape[0], msa.shape[1]])

        pf_info = np.array(pf_info)
        np.save(pf_info_file, pf_info)

    return pf, pf_info, pdbmap
