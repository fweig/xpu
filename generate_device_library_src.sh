#!/bin/bash

KERNEL_DECL_FILE=
SRCS=()
OUTFILE=
OUTPUT_TYPE=

PARSING_SRCS=false
until [ -z "$1" ]
do
    case "$1" in
        "-k")
            KERNEL_DECL_FILE="$2"
            shift 2
            ;;
        "-o")
            OUTFILE="$2"
            shift 2
            ;;
        "-s")
            PARSING_SRCS=true
            shift
            ;;
        "-t")
            OUTPUT_TYPE="$2"
            shift  2
            ;;
        *)
            if $PARSING_SRCS;
            then
                SRCS+=("$1")
                shift
            else
                warn "unknown argument, '$1'\n"
                usage
                exit 1
            fi
			;;
    esac
done


rm -f "$OUTFILE"

case "$OUTPUT_TYPE" in
    "header")
        echo "#include <xpu/device_library/template/frontend.h>" >> "$OUTFILE"
        ;;
    "frontend")
        echo "#include <xpu/device_library/template/backend.h>" >> "$OUTFILE"
        echo "#include <xpu/device_library/template/frontend.cpp>" >> "$OUTFILE"

        for src in "${SRCS[@]}"
        do
            echo "#include <$src>" >> "$OUTFILE"
        done
        ;;
    "backend")
        echo "#include <xpu/device_library/template/backend.h>" >> "$OUTFILE"
        echo "#include <xpu/device_library/template/backend.cpp>" >> "$OUTFILE"

        for src in "${SRCS[@]}"
        do
            echo "#include <$src>" >> "$OUTFILE"
        done
        ;;
    *)
        # TODO: handle error
        ;;
esac
