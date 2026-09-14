SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

all: copy-floats paper clean

copy-floats:
	@mkdir -p tables
	@cp ../results/tables/* ./tables/
	@mkdir -p figures
	@cp ../results/figures/* ./figures/
	@echo "✅ PDF compiled successfully: paper.pdf"
	@echo "✅ PDF compiled successfully: supplement.pdf"

paper: supplement
	@echo "Compiling main manuscript PDF ..."
	pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex
	@echo "✅ PDF compiled successfully: paper.pdf"
	@echo "Re-resolving cross-references in supplement.tex against updated paper.aux ..."
	pdflatex -interaction=nonstopmode supplement.tex
	@echo "✅ PDF compiled successfully: supplement.pdf"	

.PHONY: supplement
supplement:
	@echo "Compiling supplementary materials PDF ..."
	pdflatex -interaction=nonstopmode supplement.tex && pdflatex -interaction=nonstopmode supplement.tex
	@echo "✅ PDF compiled successfully: supplement.pdf"

clean:
	@rm -f _main.log _main.fls _main.fdb_latexmk _main.aux 
	@rm -fr _main.synctex.gz
	@rm -fr _main_files/
	@rm -f paper.log paper.fls paper.fdb_latexmk paper.aux 
	@rm -fr paper.synctex.gz
	@rm -f supplement.log supplement.fls supplement.fdb_latexmk supplement.aux
	@rm -fr supplement.synctex.gz
