#!/bin/sh


# =================================================================
# =================================================================

#			STEP 1. Complete Info

# =================================================================
# =================================================================


# CUDA
# FILL: load your cuda module, example: module load cuda/8.0.61

# PARAMETERES.
# FILL: shell scripts receive following parameters:
# $1. Number of nodes
# $2. Network
# $3: Blocks
# $4. Block Sizes
# $5. Batch Sizes
# $6. Dataset used


# FOLDERS.
# FILL: adapt you folders paths

# FUPERMOD_FOLDER = ...
# BIN_FOLDER=$FUPERMOD_FOLDER/bin
# LIB_FOLDER=$FUPERMOD_FOLDER/lib
# FPM_FOLDER=$FUPERMOD_FOLDER/fpm
# CONFIG_FOLDER= ... output folder ...


# PROCCESES PATTERNS:
# Examples:


#patt_21=("NODE 0 0-21 cpu OMP_NUM_THREADS=22"$subopts"
#NODE 1 22 gpu id=0"$subopts"
#NODE 2 23 gpu id=1"$subopts"")

#patt_22=("NODE 0 0-22 cpu OMP_NUM_THREADS=23"$subopts"
#NODE 1 23 gpu id=0"$subopts"")


HOSTFILE=$CONFIG_FOLDER/hosts
HOSTFILE_TMP=$CONFIG_FOLDER/tmp_hosts

scontrol show hostname ${SLURM_JOB_NODELIST} | sort > hosts.txt
scontrol show hostname ${SLURM_JOB_NODELIST} | sort > $HOSTFILE_TMP

CFILE=$CONFIG_FOLDER/M"$numNodes".conf

# Header of CFILE: from this file we generate the RANKFILE and the CONFFILE
echo -e "# Configuration: M"$numNodes  > $CFILE

# Number of procs
i=0

# This is used for assign rank of procceses in RankFile in STEP 2.
patt_22_nr=0
patt_21_nr=0

# Number of processes per pattern
PATT_NRS24=3

# Number of processes is calculated from the configuration applied.
numProcs=0





# =================================================================
# =================================================================

#		STEP 2. Building RANKFILE and CONFFILE

# =================================================================
# =================================================================





# Patterns from hostfile to a preliminar configuration file in CFILE
for line in $(cat $HOSTFILE_TMP)
do

	core_nr=0

	case "$line" in

		"Node22_Hostname" ) core_nr=;; #FILL WITH YOUR CORE NUMBERS, example is given (10)
		"Node21_Hostname" ) core_nr=;; #FILL WITH YOUR CORE NUMBERS, example is give (24)

	esac

	# Write the hostfile name and the number of slots.

	if [ "$core_nr" -eq 24 ]; then
		echo $line" slots=24"  >> $HOSTFILE
	fi
	if [ "$core_nr" -eq 10 ]; then
		echo $line" slots=10"  >> $HOSTFILE
	fi
  
	

	node_name[$i]="$line"
	let "i += 1"

	# Write the .conf file
	if [ "$core_nr" -eq "24" ]; then
		c_line=${patt_24[$patt_24_nr]//NODE/$line}
		let "patt_24_nr = (patt_24_nr + 1) % $PATT_NRS"
	fi
	if [ "$core_nr" -eq "9" ]; then
		c_line=${patt_9[$patt_9_nr]//NODE/$line}
		let "patt_9_nr = (patt_9_nr + 1) % $PATT_NRS"
	fi
	if [ "$core_nr" -eq "10" ]; then
		c_line=${patt_10[$patt_10_nr]//NODE/$line}
		let "patt_10_nr = (patt_10_nr + 1) % $PATT_NRS"
	fi



	# Write the line.
	echo -e "$c_line" >> $CFILE

done




RANKFILE=$CONFIG_FOLDER/rnk_file_conf
CONFFILE=$CONFIG_FOLDER/conf_file

# Header of RANKFILE for MPI
echo -e "# MPI RANKFILE: $RANKFILE"  > $RANKFILE

# Header of CONFFILE for FuPerMod
echo -e "# FuPerMod CONFFILE: $CONFFILE"  > $CONFFILE


# From each line of the Mx.conf file we create a line in each one of
#  the CONFFILE and RANKFILE files, with the correct format.
rank_nr=0
while IFS='' read -r line || [[ -n "$line" ]]; do

    new_line=( $line )

    # Skip or copy the comment lines
    carac=${line:0:1}
    if [ "$carac" == "#" ]; then
        echo $line >> $RANKFILE
    elif [ ! -z "$line" ]
    then

        conf_line="${new_line[0]} ${new_line[1]} ${new_line[2]} ${new_line[3]} ${new_line[4]}"
        rank_line="rank $rank_nr=${new_line[0]} slot=${new_line[2]}"

        let "rank_nr += 1"

        # Copy to files
        echo -e $conf_line >> $CONFFILE
        echo -e $rank_line >> $RANKFILE

    fi

done < $CFILE

numProcs=$rank_nr



# =================================================================
# =================================================================

#			STEP 3. FPM CALCULATION

# =================================================================
# =================================================================


cd $CONFIG_FOLDER
echo -e "  --->>  Calculating FPMs ... \n\n"


# NETWORKS for MPI parameter. FILL HERE, example is given:

if [ "$net" == "TCP" ]; then
COMMS=" tcp,self,vader "
else
COMMS=" openib,self,vader "
fi




OMPIALG=" --mca coll_tuned_priority 100 --mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_bcast_algorithm 1 "

# LIBS=" $BIN_FOLDER/bin/builder_"$kernel"_"$mode"_"$K" -l $BIN_FOLDER/lib/libwave_1d_"$kernel"_"$mode"_"$K".so "
LIBS=" $FUPERMOD_FOLDER/bin/builder -l $FUPERMOD_FOLDER/lib/libmxm_1d.so "

# Here we assign the batch size to calculate the relative speed of the processors in the benchmark.
# FILL -p with the path of launch_fpm.py and -P with config_folder path.
LIBS_OPTS=" -L$bsz -U$bsz -s1 -i0.98 -r5 -p FILL -P FILL"

mpirun -n $numProcs --rankfile $RANKFILE --mca btl $COMMS $OMPIALG --bind-to core -report-bindings --display-map --nooversubscribe $LIBS $LIBS_OPTS


if [ "$?" -eq 0 ]; then
    echo "OK"
else
    echo "FAIL"
fi




# =================================================================
# =================================================================

#		STEP 4. PARTITION OF THE DATASET

# =================================================================
# =================================================================



# Depends on k and N
# We take k as a constant (block size)
# Values of N and D are calculated

echo -e "  --->>  Calculating Partitions ... \n\n"


# FILL HERE: You need to provide the image size calculated as (batch size * width * height * canals). Example is given for CIFAR10.
let "D = $bsz * 32 * 32 * 3"

# N is the number of elements in a row or a column. Pixel definition
let "N = $blocks * $K"

echo -e "$name:  b = $blocks,  D = $D,  N = $N  K = $K"

# Partitioner.
$FUPERMOD_FOLDER/bin/partitioner -l $FUPERMOD_FOLDER/lib/libmxm_1d.so -a3 -D $D -p $CONFIG_FOLDER/part.dist


if [ "$?" -eq 0 ]; then
    echo "OK"
else
    echo "FAIL"
fi 


echo $numProcs
OMPIALG=" --mca pmix_suppress_missing_data_warning 1 "





# =================================================================
# =================================================================

#		STEP 5 (LAST). LAUNCH THE NEURAL NETWORK

# =================================================================
# =================================================================



# Give n repetitions to calculate average results. Parameters for launch_dnn.py are:
# P1. Batch Size
# P2. Balanced or not balanced (0 or 1)
# P3. $CONFIG_FOLDER/part.dist
# P4. Dataset
# P5. $reps
# P6. Type of resnet (if needed)

# Example is given: launch_dnn.py $CONFIG_FOLDER/part.dist 4096 1 $CONFIG_FOLDER/part.dist SimpleModel_CIFAR10 $reps "20"

for reps in {1..5}
do
	balanced=0
	mpirun -n $numProcs --rankfile $RANKFILE --bind-to core  -report-bindings --display-map -nooversubscribe -quiet python launch_dnn.py $CONFIG_FOLDER/part.dist $P1 $P2 $P3 $P4 $P5 $P6

	balanced=1
	mpirun -n $numProcs --rankfile $RANKFILE --bind-to core  -report-bindings --display-map -nooversubscribe -quiet python launch_dnn.py $CONFIG_FOLDER/part.dist $P1 $P2 $P3 $P4 $P5 $P6

done
