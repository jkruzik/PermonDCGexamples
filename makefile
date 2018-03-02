ALL: file

include ${PERMON_DIR}/lib/permon/conf/permon_base

file: file.o chkopts
	-${CLINKER} -o file file.o ${PERMON_LIB}
	${RM} file.o

