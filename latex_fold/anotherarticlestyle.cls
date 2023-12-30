%% Customized Nature-Style Class
%% Based on natureprintstyle.cls by Brendon Higgins, Michele Cappellari, and Grant Tremblay
%% https://github.com/granttremblay/Nature_Letter_LaTeX_template/blob/master/natureprintstyle.cls
%% Original versions and contributions: v1.0 - Brendon Higgins (19 April 2008), 
%% v1.1 - Michele Cappellari (13 February 2012), v1.2 - Grant Tremblay (25 April 2016)

%% Modified by Pere Rosselló, per.rossello@gmail.com, 10/02/2023

\ProvidesClass{anotherarticlestyle}[10/02/2023]

\LoadClass[11pt,twocolumn, twoside]{extarticle}

\AtBeginDocument{\sansmath}

\renewcommand{\rmdefault}{\sfdefault}
\RequirePackage[framemethod=TikZ]{mdframed}
\RequirePackage{cuted}
\RequirePackage{lipsum}
\RequirePackage{natbib}
\RequirePackage[bookmarks = true, colorlinks=true, linkcolor = black, citecolor = black, menucolor = black, urlcolor = black]{hyperref}
\RequirePackage{booktabs}
\RequirePackage{amsmath}
\RequirePackage{sansmath}
\RequirePackage{graphicx}
\RequirePackage{afterpage}
\RequirePackage{natbib}
\RequirePackage{cite}
\RequirePackage{titlesec}
\RequirePackage{ifthen}
\RequirePackage[a4paper, left=1.7cm, right=1.7cm, top=2.3cm, bottom=2.3cm]{geometry}
\RequirePackage{scalefnt}
\RequirePackage{type1cm}
\RequirePackage{fancyhdr}
\RequirePackage{color}
\definecolor{ullpurple}{rgb}{0.39765625, 0, 0.39765625}
\definecolor{lightpurple}{rgb}{0.99609375, 0.896875  , 0.99609375}

\pagestyle{fancy}
\fancyhf{} % Clear all headers and footers first
\fancyfoot[LE,RO]{\textcolor{ullpurple}{\textbf{\thepage}}} % Left on even pages, Right on odd pages
\renewcommand{\headrulewidth}{0.3pt} % Top rule
\renewcommand{\footrulewidth}{0.3pt} % Bottom rule
\renewcommand{\headrule}{\hbox to\headwidth{\color{gray}\leaders\hrule height \headrulewidth\hfill}}
\renewcommand{\footrule}{\hbox to\headwidth{\color{gray}\leaders\hrule height \footrulewidth\hfill}}
\newcommand{\leftheader}[1]{\fancyhead[LE]{\vspace{-12pt}\textcolor{gray}{#1}}}
\newcommand{\rightheader}[1]{\fancyhead[RO]{\vspace{-12pt}\textcolor{gray}{#1}}}



%% Some style parameters
\setlength{\parindent}{0.2in}
\newcommand{\spacing}[1]{\renewcommand{\baselinestretch}{#1}\large\normalsize}

\newcommand{\header}[1]{\def\@header{#1}}

\def\@maketitle{%
  \newpage\spacing{1.0}\setlength{\parskip}{-10pt}%
  \begin{minipage}[b]{0.8\linewidth}
    {\color{ullpurple} \fontsize{20}{10}\selectfont \@header \par \color{black}}%
  \end{minipage}%
  \hfill
  \begin{minipage}[b]{0.2\linewidth}
    \raisebox{-0.2\height}{\includegraphics[width=\linewidth]{logos/logo_ull.jpg}}
  \end{minipage}%
  \par
  \rule{\textwidth}{0.03pt}
  {\scalefont{1.7}\noindent\sloppy%
    \begin{flushleft}\bfseries\@title\end{flushleft} \par \vspace{0.25cm}}%
  {\scalefont{1.0}\noindent\sloppy  \@author \par \vspace{-0.6cm}}%
}



\newenvironment{affiliation}{%
  %    \begin{strip}%
  \let\olditem=\item%
  \renewcommand\item[1][]{$^{\textcolor{gray}{\arabic{enumi}}}$\stepcounter{enumi}}%
  \setcounter{enumi}{1}%
  \setlength{\parindent}{0in}%
  \sffamily\sloppy%
  \scalefont{0.83}%
}{%
  \let\item=\olditem%
  %    \end{strip}%
}



\newcounter{contactSymbolCounter}
\newcommand{\nextContactSymbol}{%
  \stepcounter{contactSymbolCounter}%
  \ifcase\thecontactSymbolCounter\or\textcolor{gray}{$^{*}$}\or\textcolor{gray}{$^\dagger$}\or\textcolor{gray}{$^{\ddagger}$}\else\textcolor{gray}{$^{??}$}\fi%
}



\newenvironment{contact}{%
  \let\olditem=\item%
  \setcounter{contactSymbolCounter}{0}%
  %\renewcommand\item{\nextContactSymbol}%
  \renewcommand\item{\hspace{0.1in}\nextContactSymbol}
  \setlength{\parindent}{0.0in}%
  \sffamily\sloppy%
  \scalefont{0.83}%
}{%
  \let\item=\olditem%
}

\newcommand{\correspondence}[1]{\def\theCorrespondence{#1}}
\newcommand{\affiliations}[1]{\def\theAffiliations{#1}}


\fancypagestyle{plain}{%
  \fancyhf{} % Clear all headers and footers first
  \fancyfoot[L]{%
    \vspace{-30pt}
    \color{gray}{\rule{\textwidth}{0.03pt}} \\ % The rule
    \begin{affiliation}
      \item \textcolor{gray}{\theAffiliations}
    \end{affiliation}
    \newline
    \begin{contact}
      \item \textcolor{gray}{\theCorrespondence }
    \end{contact}
  }
  \renewcommand{\headrulewidth}{0pt} % No header line
  \renewcommand{\footrulewidth}{0pt} % No footer line (since you're manually adding the rule)
}




\renewenvironment{abstract}{%
  \begin{mdframed}[backgroundcolor=lightpurple,linewidth=0pt]
    \setlength{\parindent}{0in}%
    \setlength{\parskip}{0.2in}%
    %\bfseries%\sffamily\scalefont{1}%
    }{%
    \vspace{\baselineskip}
  \end{mdframed}%
  \par
}

\makeatletter
\def\@seccntformat#1{%
  \ifcsname the#1\endcsname
    \csname the#1\endcsname\hspace{0.5em}%
  \fi
}

\renewcommand{\labelitemi}{{\color{ullpurple}\textbullet}}


\titleformat{\section}
{\sffamily\bfseries\scalefont{1.3}\color{ullpurple}}
{\thesection}
{0.5em}
{}

\titleformat{\subsection}
{\sffamily\bfseries\scalefont{1}\color{ullpurple}}
{\thesubsection}
{0.5em}
{}

\titleformat{\subsubsection}
{\sffamily\bfseries\scalefont{1}\color{ullpurple}}
{\thesubsubsection}
{0.5em}
{}
% \renewcommand{\section}{\@startsection {section}{1}{-0.2in}%
%   {12pt}{2pt}%
%   % {\ifnum\value{section}>0%
%   %     \sffamily\bfseries\scalefont{1.3}\textcolor{ullpurple}%
%   %   \else
%   %     \sffamily\bfseries\scalefont{1.3}%
%   %   \fi
%   % }%
% }

\makeatother


%% No heading for References section, but eat up the extra space from \section command
\renewcommand\refname{}




%% Define the addendum environment for Supplementary Info, Acknowledgements, etc.
\newenvironment{addendum}{%
  \setlength{\parindent}{0in}%
  \sffamily%
  \scalefont{0.83}%
  \begin{list}{Acknowledgements}{%
      \setlength{\leftmargin}{0in}%
      \setlength{\listparindent}{0in}%
      \setlength{\labelsep}{0em}%
      \setlength{\labelwidth}{0in}%
      \setlength{\itemsep}{5pt}%
      \let\makelabel\addendumlabel}
    }
    {\end{list}\par\vfill}

\newcommand*{\addendumlabel}[1]{\textbf{#1}\hspace{1em}}

%% Figures and tables:
%% The \includegraphics command is respected.
%% Tables should all be moved to the end of the document
%% manually.

\renewcommand{\figurename}{Figure}
\renewcommand{\tablename}{Table}

%% Change the way captions are formatted. Based on a tip by Jakob Schiotz.
%% http://dcwww.camd.dtu.dk/~schiotz/comp/LatexTips/LatexTips.html
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{{\textbf{\scalefont{0.93}#1 $\vert$}} \scalefont{0.93}#2}%
  \ifdim \wd\@tempboxa >\hsize
    {\scalefont{0.93}{\textbf{\scalefont{0.93}#1 $\vert$}} \scalefont{0.93}#2\par}
  \else
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
