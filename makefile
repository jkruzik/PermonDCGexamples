ALL: ex2

include ${PERMON_DIR}/lib/permon/conf/permon_base

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${PERMON_LIB}
	${RM} ex2.o

file: file.o chkopts
	-${CLINKER} -o file file.o ${PERMON_LIB}
	${RM} file.o

