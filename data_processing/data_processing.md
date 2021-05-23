<!-- MarkdownTOC -->

- [ctc_to_mot](#ctc_to_mo_t_)
- [reformatMOT](#reformatmot_)
    - [CTMC       @ reformatMOT](#ctmc___reformatmo_t_)
        - [train       @ CTMC/reformatMOT](#train___ctmc_reformatmot_)
        - [test       @ CTMC/reformatMOT](#test___ctmc_reformatmot_)

<!-- /MarkdownTOC -->

<a id="ctc_to_mo_t_"></a>
# ctc_to_mot
python3 ctc_to_mot.py

<a id="reformatmot_"></a>
# reformatMOT

<a id="ctmc___reformatmo_t_"></a>
## CTMC       @ reformatMOT-->data_processing

<a id="train___ctmc_reformatmot_"></a>
### train       @ CTMC/reformatMOT-->data_processing

python3 reformatMOT.py root_dir=/data/CTMC db_type=train process_tra=0 ignore_det=1 start_out_id=0 no_move=0
python3 reformatMOT.py root_dir=/data/CTMC db_type=train process_tra=1 ignore_det=1 start_out_id=0 no_move=0

<a id="test___ctmc_reformatmot_"></a>
### test       @ CTMC/reformatMOT-->data_processing

python3 reformatMOT.py root_dir=/data/CTMC db_type=test process_tra=0 ignore_det=1 start_out_id=0 no_move=0
python3 reformatMOT.py root_dir=/data/CTMC db_type=test process_tra=1 ignore_det=1 start_out_id=0 no_move=0