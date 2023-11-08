"""

Frecventele emise de un contrabass se incadreaza intre 40Hz si 200Hz. Care
este frecventa minima cu care trebuie esantionat semnalul trece-banda
provenit din inregistrarea instrumentului, astfel incat semnalul discretizat
sa contina toate componentele de frecventa pe care instrumentul le poate
produce?

R: Conform teoremei Nyquist-Shannon, din fisa laboratorului,
frecventa minima de esantionare (fs) si frecventa maxima continuta in semanl (B)
au o relatie pe baza careia se poate afla fs < = > fs > 2B.
Cu alte cuvinte, B in cazul nostru este 200Hz deci fs > 2 * 200
fs > 400Hz <=> orice frecventa mai mare decat 400Hz este buna

"""