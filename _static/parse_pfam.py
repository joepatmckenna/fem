import pickle
import numpy as np
import pandas as pd
import Bio
from Bio import AlignIO
import os, urllib, gzip, re


def parse_pfam(data_dir='./'):

    # download pfam files from database ftp
    # ftp faq: https://pfam.xfam.org/help#tabview=tab13
    pfam_current_release = 'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release'

    files = [
        'Pfam-A.full.gz',  # The full alignments of the curated families (~6GB)
        # 'pdbmap.gz',  # Mapping between PDB structures and Pfam domains. (~2MB)
    ]
    for f in files:
        local = os.path.join(data_dir, f)
        if not os.path.exists(local):
            remote = os.path.join(pfam_current_release, f)
            urllib.urlretrieve(remote, local)

    # get list of
    # 0) accession codes,
    # 1) number of residues,
    # 2) number of sequences,
    # 3) and number of pdb references
    # for all pfams in database
    pfam_file = os.path.join(data_dir, 'pfam.npy')
    if os.path.exists(pfam_file):
        pfam = np.load(pfam_file)
    else:
        pfam = []
        with gzip.open(os.path.join(data_dir, 'Pfam-A.full.gz'), 'r') as f:
            for i, line in enumerate(f):
                if line[:7] == '#=GF AC':
                    ac = line.split(' ')[4][:-1].split('.')[0]
                    pfam.append([ac, 0, 0, 0])

        pfam = np.array(pfam)
        n_pfam = len(pfam)

        # parse multiple sequence alignments file with Biopython
        alignments = AlignIO.parse(
            gzip.open(os.path.join(data_dir, 'Pfam-A.full.gz'), 'r'),
            'stockholm')

        # for each pfam/msa
        for i, a in enumerate(alignments):

            # local directory associated with pfam
            pfam_dir = os.path.join(data_dir, 'Pfam-A.full', pfam[i, 0])

            try:
                os.makedirs(pfam_dir)
            except:
                pass

            # number of residues/sequences in alignment
            n_residue = a.get_alignment_length()
            n_sequence = len(a)

            # msa: residues symbols
            # pdb_refs: references to pdb
            msa = np.empty((n_residue, n_sequence), dtype=str)
            pdb_refs = []

            # for each sequence in alignment
            for j, seq in enumerate(a):
                # store residue symbols in lowercase
                msa[:, j] = np.array(seq.seq.lower())
                # store uniprot sequence id
                uniprot_id, uniprot_start, uniprot_end = re.split(
                    '[/-]', seq.id)
                # extract pdb refs if they are present
                refs = seq.dbxrefs
                if not refs:
                    continue
                for ref in refs:
                    ref = ref.replace(' ', '').split(';')
                    if ref[0] == 'PDB':
                        pdb_id, chain = ref[1][:-1], ref[1][-1]
                        pdb_start_end = ref[2].split('-')
                        if len(pdb_start_end) == 2:
                            pdb_start, pdb_end = pdb_start_end
                        else:
                            continue
                        pdb_refs.append([
                            pfam[i, 0], j - 1, uniprot_id, uniprot_start,
                            uniprot_end, pdb_id, chain, pdb_start, pdb_end
                        ])

            np.save(os.path.join(pfam_dir, 'msa.npy'), msa)
            np.save(os.path.join(pfam_dir, 'pdb_refs.npy'), pdb_refs)

            n_pdb_ref = len(pdb_refs)

            pfam[i, 1:] = n_residue, n_sequence, n_pdb_ref

        np.save(pfam_file, pfam)

    # compile all pdb references
    pdb_refs_file = os.path.join(data_dir, 'pdb_refs.npy')
    if os.path.exists(pdb_refs_file):
        pdb_refs = np.load(pdb_refs_file)
    else:
        n_pdb_refs = pfam[:, 3].astype(int)
        has_pdb_refs = n_pdb_refs > 0
        n_pdb_refs = n_pdb_refs[has_pdb_refs]
        pfam_with_pdb = pfam[has_pdb_refs, 0]
        refs = os.path.join(data_dir, 'Pfam-A.full', pfam_with_pdb[0],
                            'pdb_refs.npy')
        pdb_refs = np.load(refs)
        for fam in pfam_with_pdb[1:]:
            refs = np.load(
                os.path.join(data_dir, 'Pfam-A.full', fam, 'pdb_refs.npy'))
            pdb_refs = np.vstack([pdb_refs, refs])
        np.save(pdb_refs_file, pdb_refs)

    # convert pfam and pdb_refs to pandas dataframes
    pfam = pd.DataFrame(
        index=pfam[:, 0],
        data=pfam[:, 1:],
        columns=['res', 'seq', 'pdb_refs'],
        dtype=int)
    pdb_refs = pd.DataFrame(
        index=pdb_refs[:, 0],
        data=pdb_refs[:, 1:],
        columns=[
            'seq', 'uniprot_id', 'uniprot_start', 'uniprot_end', 'pdb_id',
            'chain', 'pdb_start', 'pdb_end'
        ])
    cols = ['seq', 'uniprot_start', 'uniprot_end', 'pdb_start', 'pdb_end']
    pdb_refs[cols] = pdb_refs[cols].apply(pd.to_numeric)

    # remove rows where the length between uniprot start/end is different from pdb start/end
    cols = ['uniprot_start', 'uniprot_end', 'pdb_start', 'pdb_end']
    start_end = pdb_refs[cols].values
    consistent_length = np.diff(
        start_end[:, :2], axis=1) == np.diff(
            start_end[:, 2:], axis=1)
    pdb_refs = pdb_refs[consistent_length]

    return pfam, pdb_refs

    # pdb_map_file = os.path.join(data_dir, 'pdbmap.gz')
    # names = [
    #     'pdb_id', 'chain', 'lig', 'name', 'pfam', 'pfam_protein_id', 'res'
    # ]
    # pdb_map = pd.read_csv(
    #     pdb_map_file,
    #     sep='\t',
    #     engine='python',
    #     header=None,
    #     names=names,
    #     dtype=str,
    #     compression='gzip')
    # for name in names:
    #     pdb_map[name] = pdb_map[name].map(lambda x: x.rstrip(';'))
    # pdb_map.set_index('pdb_id', inplace=True)

    # pdb_map_pfam = pdb_map['pfam'].unique()
