PYTHON := python3
SECS := secs
APXS := apxs
# NOTEBOOKS := $(wildcard $(SECS)/*.ipynb)
NOTEBOOKS := $(filter-out $(SECS)/%BACKUP%.ipynb, $(wildcard $(SECS)/0*.ipynb))
QMDS := $(NOTEBOOKS:.ipynb=.qmd)

secs: $(QMDS)

.PHONY: FORCE #%.qmd
FORCE:

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# %.qmd: %.ipynb FORCE
# 	echo "quarto convert $< -> $@ (strip YAML; split at '# APPENDIX')"
# 	quarto convert "$<" -o "$@.tmp"
# 	# Remove first 4 lines (your fixed-size YAML header)
# 	tail -n +5 "$@.tmp" > "$@.body"
# 	# Remove Colab metadata (id, colab, outputId)
# 	sed -i '' -E '/^#\| (id|colab|outputId):/d' "$@.body"

# 	# Derive appendix path: <dir>/apx-<basename>
# 	if csplit -f "$@.part" -n 1 -s "$@.body" '/^\# APPENDIX/'; \
# 	then \
# 		tail -n +2 "$@.part1" > "$(APXS)/$(@F)"; \
# 		rm "$@.part1"; \
# 		mv "$@.part0" "$@"; \
# 	else \
# 		mv "$@.body" "$@"; \
# 	fi
# 	# Remove stale appendix if it wasn't created / is empty
# 	rm -f "$@.tmp" "$@.body"


# preview:
# 	quarto preview _main.qmd --port 4444 --no-browser --no-watch-inputs

# render: 
# 	quarto render _main.qmd --execute --wrap=preserve

clean:
	rm -f _main.log _main.fls _main.fdb_latexmk _main.aux 
	rm -fr _main.synctex.gz
	rm -fr _main_files/
	rm paper.log paper.fls paper.fdb_latexmk paper.aux 
	rm -fr paper.synctex.gz

# paper: secs render


# wordcount:
# 	@/opt/homebrew/bin/pdftotext -nopgbrk -f 2 _main.pdf tmp.txt
# 	@if csplit -s tmp.txt '/^Supplemental Materials$$/'; then mv xx00 tmp.txt; rm -f xx0*; fi
# 	@sed -i '' '/^[0-9]+$$/d' tmp.txt
# 	@WORDCOUNT=$$(wc -w < tmp.txt | tr -d ' '); \
# 	sed -i '' "s/^wordcount: .*/wordcount: $$WORDCOUNT/" _main.qmd; \
# 	echo "number of words: $$WORDCOUNT"
# 	@rm -f tmp.txt

# extract-content:
# 	@echo "Extracting content from \\section{Introduction} to \\section{References}..."
# 	@sed -n '/\\\section{Introduction}/,/\\section{References}/{ /\\section{References}/!p; }' _main.tex > main_content.tex
# 	@echo "✅ Content extracted to main_content.tex"

paper:
	@echo "Compiling PDF ..."
	pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex
	@echo "✅ PDF compiled successfully: paper.pdf"

PAGE ?= 40
separate-paper:
	@if [ -z "$(PAGE)" ]; then \
		echo "Usage: make separate PAGE=<page_number>"; \
		exit 1; \
	fi
	@/opt/homebrew/bin/qpdf paper.pdf --pages paper.pdf 1-$(PAGE) -- manuscript.pdf
	@/opt/homebrew/bin/qpdf paper.pdf --pages paper.pdf $$(($(PAGE) + 1))-z -- sm.pdf
	@cp manuscript.pdf ../paper/submissions/pa/final/
	@cp sm.pdf ../paper/submissions/pa/final/
	@echo "✅ Separate PDFs created successfully: manuscript.pdf and sm.pdf"


# arXiv preparation and testing
arxiv-prepare:
	python3 ./prepare_arxiv.py --overwrite

# arxiv-compile: arxiv-prepare
# 	@echo "Compiling arXiv version..."
# 	cd _arxiv && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
# 	@echo "✅ arXiv PDF compiled successfully: _arxiv/main.pdf"

# arxiv-clean:
# 	@echo "Cleaning arXiv auxiliary files..."
# 	cd _arxiv && rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.ptc *.fdb_latexmk *.synctex.gz
# 	@echo "✅ arXiv auxiliary files cleaned"

# arxiv: arxiv-compile arxiv-clean

# arxiv-zip: arxiv-prepare
# 	@echo "Creating arXiv submission zip file..."
# 	cd _arxiv && zip -r ../arxiv-submission.zip . -x "*.aux" "*.log" "*.bbl" "*.blg" "*.out" "*.toc" "*.ptc" "*.lof" "*.lot" "*.fls" "*.fdb_latexmk" "*.synctex.gz" "main.pdf"
# 	@echo "✅ arXiv submission zip created: arxiv-submission.zip"

# # Builds the CPU analysis image; full output is both shown live and saved to
# # build.log so issues can be shared/diagnosed without re-running the build.
# docker-build:
# 	docker build -f Dockerfile -t $(IMAGE_TAG) . 2>&1 | tee build.log

# docker-gpu-build:
# 	docker build -f Dockerfile-GPU -t $(GPU_IMAGE_TAG) . 2>&1 | tee build-gpu.log

# # Builds (if needed) then runs the CPU image with data/results mounted back
# # to the host; full output is saved to run.log.
# docker-run: docker-build
# 	docker run --rm \
# 		-v "$$(pwd)/data:/project/data:ro" \
# 		-v "$$(pwd)/results:/project/results" \
# 		$(IMAGE_TAG) 2>&1 | tee run.log

# # Builds (if needed) then runs the GPU image with GPU access and
# # data/results mounted back to the host; full output is saved to gpu-run.log.
# docker-gpu-run: docker-gpu-build
# 	docker run --rm --gpus all \
# 		-v "$$HOME/.cache/huggingface:/root/.cache/huggingface" \
# 		-v "$$(pwd)/data:/project/data:ro" \
# 		-v "$$(pwd)/results:/project/results" \
# 		$(GPU_IMAGE_TAG) 2>&1 | tee gpu-run.log

# # Builds (if needed) then runs the selected GPU smoke tests.
# docker-gpu-run-test: docker-gpu-build
# 	docker run --rm --gpus all \
# 		-v "$$HOME/.cache/huggingface:/root/.cache/huggingface" \
# 		-v "$$(pwd)/data:/project/data:ro" \
# 		-v "$$(pwd)/../data/ravdess/raw/videos:/project/data/ravdess/videos:ro" \
# 		-v "$$(pwd)/results:/project/results" \
# 		$(GPU_IMAGE_TAG) \
# 		bash -c 'cd /project/code/ravdess/03-run_experiments && bash run_test.sh' \
# 		2>&1 | tee gpu-run-test.log


# Default notebooks to export; override on the command line with NOTEBOOKS=...
REPLICATION_NOTEBOOKS ?= \
	code/04-application.ipynb \
	code/05-analyses.ipynb \
	code/create_cmp_translations_dataset.ipynb \
	code/create_dataset.ipynb\


python-scripts:
	@if [ -z "$(REPLICATION_NOTEBOOKS)" ]; then \
		echo "Usage: make python-scripts ...\""; \
		exit 1; \
	fi
	@for nb in $(REPLICATION_NOTEBOOKS); do \
		out="$${nb%.ipynb}.py"; \
		echo "Exporting $$nb -> $$out"; \
		mkdir -p "$$(dirname "$$out")"; \
		$(PYTHON) -m nbconvert --to python --output "$$(basename "$$out")" --output-dir "$$(dirname "$$out")" "$$nb"; \
		sed -E -i '' \
			-e "s|Path\(\*\['\.\.'\]\*2\)|Path(__file__).resolve().parent / Path(*['..']*2)|g" \
			"$$out"; \
		sed -E -i '' \
			-e 's/^([[:space:]]*)plt\.show\(\)/\1# plt.show()/g' \
			-e 's/^([[:space:]]*)print\(/\1# print(/g' \
			"$$out"; \
		sed -E -i '' \
			-e '/^[[:space:]]*# In\[[^]]*\]:/d' \
			-e '/^[[:space:]]*# %%/d' \
			-e 's/^# (#+.*)$$/\1/' "$$out"; \
	done