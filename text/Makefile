DOCNAME= Petkovic_Stamenic_podesavanje_tezina_neuronske_mreze
all: sem

.PHONY: clean

sem:
	pdflatex $(DOCNAME).tex
	bibtex $(DOCNAME).aux 
	pdflatex $(DOCNAME).tex
	pdflatex $(DOCNAME).tex

view: sem
	okular $(DOCNAME).pdf &

clean:
	rm *.blg *.bbl *.aux *.log *.out *.toc *.pdf
