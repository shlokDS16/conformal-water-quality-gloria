"""Reference checker for paper/refs.bib and paper/main.tex (phase 12, CLAUDE.md section 2).

For every BibTeX entry:
  * resolve the DOI on Crossref (https://api.crossref.org/works/<doi>), falling back to DataCite
    (https://api.datacite.org/dois/<doi>) for dataset, Zenodo and arXiv DOIs;
  * compare title (fuzzy), first-author surname, year, journal or venue, volume and pages;
  * entries without a DOI: search Crossref by title (and, when reachable, Semantic Scholar) and
    report a candidate DOI; when no DOI exists, require a URL and record it;
  * flag uncited entries and citations without entries;
  * flag preprints (arXiv, bioRxiv, SSRN) that are not marked as preprints, and report whether a
    peer-reviewed version now exists;
  * flag retracted works (Crossref `update-to` / `relation.is-retracted-by`).

Writes research/REFERENCE_AUDIT.md (metadata section only; the claim-support and coverage sections
of that file are maintained by hand and are preserved across runs) and exits non-zero on any
failure.

Usage:
    python src/check_references.py [--bib paper/refs.bib] [--tex paper/main.tex]
                                   [--out research/REFERENCE_AUDIT.md]
                                   [--cache research/.refcache.json] [--refresh] [--offline]

Network etiquette: every Crossref call carries the polite-pool `mailto` parameter and a descriptive
User-Agent; requests are serialised with a fixed delay and retried with exponential backoff on
HTTP 429 and 5xx. OpenAlex is queried only as a last resort (it has been rate limiting) and a
failure there is never fatal.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MAILTO = "shlok.goenka2023@vitstudent.ac.in"
UA = f"check_references.py (https://github.com/; mailto:{MAILTO})"
DELAY = 1.0          # seconds between API calls
MAX_RETRY = 4
TITLE_MIN_RATIO = 0.88
PREPRINT_HOSTS = ("arxiv", "biorxiv", "medrxiv", "ssrn", "essoar", "preprints.org", "10.48550",
                  "10.1101", "10.2139")

# ----------------------------------------------------------------------------- bib parsing

ENTRY_RE = re.compile(r"@(\w+)\s*(\{)\s*([^,\s]+)\s*,", re.I)


def read_text(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def strip_bib_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        out.append("" if line.lstrip().startswith("%") else line)
    return "\n".join(out)


def parse_bib(text: str) -> list[dict]:
    """Brace-aware BibTeX parser; returns entries in file order."""
    text = strip_bib_comments(text)
    entries = []
    for m in ENTRY_RE.finditer(text):
        etype, key = m.group(1).lower(), m.group(3)
        i, depth = m.start(2), 0          # start at the entry's opening brace
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[m.end():i]
        fields = parse_fields(body)
        fields["_type"] = etype
        fields["_key"] = key
        entries.append(fields)
    return entries


def parse_fields(body: str) -> dict:
    fields, i, n = {}, 0, len(body)
    while i < n:
        m = re.compile(r"\s*([A-Za-z][\w-]*)\s*=\s*").match(body, i)
        if not m:
            break
        name = m.group(1).lower()
        j = m.end()
        if j < n and body[j] == "{":
            depth, k = 0, j
            while k < n:
                if body[k] == "{":
                    depth += 1
                elif body[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            value = body[j + 1:k]
            i = k + 1
        elif j < n and body[j] == '"':
            k = body.find('"', j + 1)
            value = body[j + 1:k]
            i = k + 1
        else:
            k = j
            while k < n and body[k] not in ",\n":
                k += 1
            value = body[j:k].strip()
            i = k
        fields[name] = value
        while i < n and body[i] in ", \t\r\n":
            i += 1
    return fields


# ----------------------------------------------------------------------------- normalisation

ACCENT_CMDS = re.compile(r"\\[`'^\"~=.uvHtcdbr]\s*\{?\\?([A-Za-z])\}?")
LATEX_LIG = {r"\&": "&", r"\%": "%", r"\_": "_", r"\$": "$", r"\#": "#",
             r"{\l}": "l", r"{\o}": "o", r"{\O}": "O", r"{\ss}": "ss", r"{\aa}": "aa",
             r"\ ": " ", "--": "-", "---": "-"}


def delatex(s: str) -> str:
    if not s:
        return ""
    s = ACCENT_CMDS.sub(r"\1", s)
    for a, b in LATEX_LIG.items():
        s = s.replace(a, b)
    s = re.sub(r"\\[A-Za-z]+\s*", " ", s)
    s = s.replace("{", "").replace("}", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


def norm_title(s: str) -> str:
    s = delatex(s).lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def first_surname_bib(author_field: str) -> str:
    if not author_field:
        return ""
    first = re.split(r"\s+and\s+", author_field.strip())[0].strip()
    first = delatex(first)
    if "," in first:
        return first.split(",")[0].strip()
    parts = first.split()
    return parts[-1] if parts else ""


def bib_surnames(author_field: str) -> list[str]:
    """Every surname in a BibTeX author field, in order."""
    out = []
    for a in re.split(r"\s+and\s+", (author_field or "").strip()):
        a = delatex(a.strip())
        if not a:
            continue
        out.append(a.split(",")[0].strip() if "," in a else (a.split()[-1] if a.split() else ""))
    return out


def norm_name(s: str) -> str:
    s = delatex(s).lower()
    return re.sub(r"[^a-z]", "", s)


def norm_pages(s: str) -> str:
    s = delatex(s or "")
    s = s.replace("\u2013", "-").replace("--", "-").replace(" ", "")
    return s.strip()


def norm_venue(s: str) -> str:
    s = norm_title(s)
    for stop in ("the ", "a ", "an "):
        if s.startswith(stop):
            s = s[len(stop):]
    return s


# ----------------------------------------------------------------------------- HTTP

class Net:
    def __init__(self, cache_path: Path, refresh: bool = False, offline: bool = False):
        self.cache_path = cache_path
        self.offline = offline
        self.cache: dict = {}
        if cache_path.exists() and not refresh:
            try:
                self.cache = json.loads(read_text(cache_path))
            except Exception:
                self.cache = {}
        self.last = 0.0
        self.errors: list[str] = []

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(json.dumps(self.cache, ensure_ascii=False, indent=0, sort_keys=True))

    def get_json(self, url: str, optional: bool = False):
        if url in self.cache:
            return self.cache[url]
        if self.offline:
            return None
        delay = DELAY
        for attempt in range(MAX_RETRY):
            wait = self.last + DELAY - time.time()
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.loads(resp.read().decode("utf-8", "replace"))
                self.last = time.time()
                self.cache[url] = data
                self.save()
                return data
            except urllib.error.HTTPError as exc:
                self.last = time.time()
                if exc.code == 404:
                    self.cache[url] = None
                    self.save()
                    return None
                if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRY - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if not optional:
                    self.errors.append(f"HTTP {exc.code} for {url}")
                return None
            except Exception as exc:                        # network / timeout / JSON
                self.last = time.time()
                if attempt < MAX_RETRY - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if not optional:
                    self.errors.append(f"{type(exc).__name__} for {url}")
                return None
        return None


# ----------------------------------------------------------------------------- record model

class Record:
    """Normalised bibliographic record from Crossref or DataCite."""

    def __init__(self, source: str):
        self.source = source
        self.doi = ""
        self.title = ""
        self.first_surname = ""
        self.surnames: list[str] = []
        self.year = ""
        self.years: list[str] = []      # every plausible year (online / print / issued)
        self.venue = ""
        self.venues: list[str] = []     # every container title, plus the publisher
        self.volume = ""
        self.issue = ""
        self.pages = ""
        self.type = ""
        self.abstract = ""
        self.retracted = False
        self.updates: list[str] = []


# Registry records that are demonstrably wrong, with the evidence. Each key maps a BibTeX key to
# the zero-based author positions where the entry deliberately departs from the resolved record.
AUTHOR_OVERRIDES = {
    # Crossref deposit for 10.1016/j.rse.2020.111768 swaps given and family for this author
    # (given "Matsushita", family "Bunkei"). The published article, and the Crossref records of
    # 10.1016/j.rse.2021.112860 and 10.1038/s41597-023-01973-y, all print "Bunkei Matsushita".
    "Balasubramanian2020": {10},
}


def crossref_record(msg: dict) -> Record:
    r = Record("Crossref")
    r.doi = (msg.get("DOI") or "").lower()
    r.title = (msg.get("title") or [""])[0]
    authors = msg.get("author") or []
    r.surnames = [(a.get("family") or a.get("name") or "") for a in authors]
    if authors:
        a0 = authors[0]
        r.first_surname = a0.get("family") or a0.get("name") or ""
    # Crossref `issued` is the earliest deposited date, which for many publishers is the
    # online-first date. A reference that carries volume, issue and pages is dated by the
    # issue, so `published-print` is preferred and every recorded year is accepted.
    for k in ("published-print", "issued", "published-online", "created"):
        p = (msg.get(k) or {}).get("date-parts") or [[]]
        if p and p[0] and p[0][0]:
            y = str(p[0][0])
            if y not in r.years:
                r.years.append(y)
    r.year = r.years[0] if r.years else ""
    ct = [c for c in (msg.get("container-title") or []) if c]
    r.venues = list(ct)
    if msg.get("publisher"):
        r.venues.append(msg["publisher"])
    r.venue = ct[0] if ct else (msg.get("publisher") or "")
    r.volume = str(msg.get("volume") or "")
    r.issue = str(msg.get("issue") or "")
    r.pages = str(msg.get("page") or "")
    r.type = msg.get("type") or ""
    r.abstract = msg.get("abstract") or ""
    for u in msg.get("update-to") or []:
        r.updates.append(f"{u.get('type')} -> {u.get('DOI')}")
        if u.get("type") in ("retraction", "withdrawal", "removal"):
            r.retracted = True
    rel = msg.get("relation") or {}
    for k in ("is-retracted-by", "is-withdrawn-by"):
        if rel.get(k):
            r.retracted = True
            r.updates.append(k)
    return r


def datacite_record(msg: dict) -> Record:
    attr = msg.get("attributes") or {}
    r = Record("DataCite")
    r.doi = (attr.get("doi") or "").lower()
    titles = attr.get("titles") or []
    r.title = (titles[0].get("title") if titles else "") or ""
    creators = attr.get("creators") or []
    for c in creators:
        nm = c.get("familyName") or c.get("name") or ""
        if "," in nm and not c.get("familyName"):
            nm = nm.split(",")[0]
        r.surnames.append(nm.strip())
    if r.surnames:
        r.first_surname = r.surnames[0]
    r.year = str(attr.get("publicationYear") or "")
    r.years = [r.year] if r.year else []
    r.venue = attr.get("publisher") or ""
    if isinstance(r.venue, dict):
        r.venue = r.venue.get("name") or ""
    r.venues = [r.venue] if r.venue else []
    ci = attr.get("container") or {}
    r.volume = str(ci.get("volume") or "")
    r.issue = str(ci.get("issue") or "")
    fp, lp = ci.get("firstPage"), ci.get("lastPage")
    r.pages = f"{fp}-{lp}" if fp and lp else (str(fp) if fp else "")
    types = attr.get("types") or {}
    r.type = types.get("resourceTypeGeneral") or types.get("resourceType") or ""
    descs = attr.get("descriptions") or []
    for d in descs:
        if d.get("descriptionType") == "Abstract":
            r.abstract = d.get("description") or ""
            break
    return r


def fetch_record(net: Net, doi: str) -> Record | None:
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    q = urllib.parse.quote(doi, safe="")
    data = net.get_json(f"https://api.crossref.org/works/{q}?mailto={MAILTO}", optional=True)
    if data and data.get("message"):
        return crossref_record(data["message"])
    data = net.get_json(f"https://api.datacite.org/dois/{q}", optional=True)
    if data and data.get("data"):
        return datacite_record(data["data"])
    return None


def crossref_title_search(net: Net, title: str, author: str = "") -> Record | None:
    params = {"query.bibliographic": delatex(title), "rows": "5", "mailto": MAILTO}
    if author:
        params["query.author"] = delatex(author)
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    data = net.get_json(url, optional=True)
    items = ((data or {}).get("message") or {}).get("items") or []
    best, best_r = None, 0.0
    for it in items:
        cand = crossref_record(it)
        ratio = title_ratio(title, cand.title)
        if ratio > best_r:
            best, best_r = cand, ratio
    if best and best_r >= TITLE_MIN_RATIO:
        return best
    return None


def check_url(net: Net, url: str) -> tuple[bool, str]:
    """Verify a web reference: HTTP 200 and, for HTML pages, a title we can report."""
    key = "URLCHECK " + url
    if key in net.cache:
        ok, detail = net.cache[key]
        return bool(ok), detail
    if net.offline:
        return True, "not checked (offline)"
    delay = DELAY
    for attempt in range(MAX_RETRY):
        wait = net.last + DELAY - time.time()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                ctype = resp.headers.get("Content-Type", "")
                body = b""
                if "html" in ctype.lower():
                    body = resp.read(200000)
                else:
                    resp.read(1)
                code = resp.getcode()
            net.last = time.time()
            detail = f"HTTP {code}"
            if body:
                m = re.search(rb"<title[^>]*>(.*?)</title>", body, re.S | re.I)
                if m:
                    t = re.sub(r"\s+", " ", m.group(1).decode("utf-8", "replace")).strip()
                    detail += f'; landing-page title "{t[:110]}"'
            else:
                detail += f"; {ctype or 'binary'}"
            net.cache[key] = [True, detail]
            net.save()
            return True, detail
        except urllib.error.HTTPError as exc:
            net.last = time.time()
            if exc.code in (429, 500, 502, 503) and attempt < MAX_RETRY - 1:
                time.sleep(delay)
                delay *= 2
                continue
            net.cache[key] = [False, f"HTTP {exc.code}"]
            net.save()
            return False, f"HTTP {exc.code}"
        except Exception as exc:
            net.last = time.time()
            if attempt < MAX_RETRY - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return False, type(exc).__name__
    return False, "unreachable"


def s2_lookup(net: Net, title: str) -> dict | None:
    params = {"query": delatex(title), "limit": "3",
              "fields": "title,year,externalIds,venue,abstract,publicationTypes"}
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
    data = net.get_json(url, optional=True)
    for it in (data or {}).get("data") or []:
        if title_ratio(title, it.get("title") or "") >= TITLE_MIN_RATIO:
            return it
    return None


# ----------------------------------------------------------------------------- tex parsing

CITE_RE = re.compile(r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar|"
                     r"parencite|textcite|nocite)\*?\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")


def cited_keys(tex: str) -> dict[str, int]:
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    counts: dict[str, int] = {}
    for m in CITE_RE.finditer(tex):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                counts[k] = counts.get(k, 0) + 1
    return counts


# ----------------------------------------------------------------------------- checks

def is_preprint_entry(e: dict) -> bool:
    blob = " ".join(str(e.get(f, "")) for f in ("doi", "journal", "howpublished", "url",
                                                "note", "series", "publisher")).lower()
    return any(h in blob for h in PREPRINT_HOSTS)


def marked_preprint(e: dict) -> bool:
    blob = " ".join(str(e.get(f, "")) for f in ("howpublished", "note", "journal",
                                                "series", "publisher")).lower()
    return ("preprint" in blob or "arxiv" in blob or "biorxiv" in blob or "medrxiv" in blob
            or "ssrn" in blob)


def check_entry(net: Net, e: dict) -> tuple[list[str], list[str], Record | None]:
    """Return (failures, notes, record)."""
    fails: list[str] = []
    notes: list[str] = []
    key = e["_key"]
    doi = (e.get("doi") or "").strip()
    rec = None

    if doi:
        rec = fetch_record(net, doi)
        if rec is None:
            fails.append(f"DOI {doi} did not resolve on Crossref or DataCite")
        else:
            notes.append(f"resolved on {rec.source}")
            if doi.lower().startswith("http") or doi.lower().startswith("doi:"):
                fails.append("doi field must be the bare DOI, not a URL")
            ratio = title_ratio(e.get("title", ""), rec.title)
            if ratio < TITLE_MIN_RATIO:
                fails.append(f"title mismatch (ratio {ratio:.2f}); record: {rec.title!r}")
            bib_sn, rec_sn = first_surname_bib(e.get("author", "")), rec.first_surname
            if rec_sn and norm_name(bib_sn) != norm_name(rec_sn):
                fails.append(f"first author surname {bib_sn!r} vs record {rec_sn!r}")
            # full author list: truncation, order and surname spelling
            bibs, recs = bib_surnames(e.get("author", "")), rec.surnames
            if recs:
                if len(bibs) != len(recs):
                    fails.append(f"author list has {len(bibs)} names; record has {len(recs)}")
                skip = AUTHOR_OVERRIDES.get(key, set())
                bad = [f"[{i}] {bibs[i]!r} vs {recs[i]!r}"
                       for i in range(min(len(bibs), len(recs)))
                       if i not in skip and norm_name(bibs[i]) != norm_name(recs[i])]
                if bad:
                    fails.append("author surnames differ from the record: " + "; ".join(bad[:6]))
                elif skip:
                    notes.append("author positions " + ", ".join(str(i) for i in sorted(skip))
                                 + " deliberately depart from the record (see AUTHOR_OVERRIDES)")
            if rec.years and e.get("year"):
                by = str(e["year"]).strip()
                if by not in rec.years:
                    fails.append(f"year {by} vs record {'/'.join(rec.years)}")
                elif len(set(rec.years)) > 1:
                    notes.append(f"year {by} matches one of several dates in the record "
                                 f"(print/issued/online: {'/'.join(rec.years)}); the entry uses "
                                 f"the year of the issue that carries the printed pages")
            bib_venues = [e.get(f, "") for f in ("journal", "booktitle", "series",
                                                 "publisher", "howpublished") if e.get(f)]
            if rec.venues and bib_venues:
                vr = max(difflib.SequenceMatcher(None, norm_venue(b), norm_venue(v)).ratio()
                         for b in bib_venues for v in rec.venues)
                if vr < 0.80:
                    fails.append(f"venue {bib_venues[0]!r} vs record {rec.venue!r}")
            elif rec.venues and not bib_venues:
                fails.append(f"no venue field; record has {rec.venue!r}")
            if rec.volume and e.get("volume") and str(e["volume"]).strip() != rec.volume:
                fails.append(f"volume {e['volume']} vs record {rec.volume}")
            if rec.volume and not e.get("volume") and e["_type"] == "article":
                notes.append(f"record has volume {rec.volume}, entry has none")
            if rec.pages:
                if e.get("pages"):
                    if norm_pages(e["pages"]) != norm_pages(rec.pages):
                        fails.append(f"pages {e['pages']} vs record {rec.pages}")
                else:
                    fails.append(f"no pages field; record has {rec.pages}")
            if rec.retracted:
                fails.append("RETRACTED / withdrawn according to Crossref: "
                             + "; ".join(rec.updates))
            elif rec.updates:
                notes.append("Crossref update-to: " + "; ".join(rec.updates))
    else:
        # No DOI. Search Crossref, then Semantic Scholar. A DOI-less entry is a failure
        # unless it carries a URL that resolves (the case the Guide allows for conference
        # proceedings and web references); any candidate DOI found is reported as a note so
        # the lead can decide whether an archival version should replace the entry.
        cand = crossref_title_search(net, e.get("title", ""), e.get("author", ""))
        cand_doi, cand_desc = "", ""
        if cand and cand.doi:
            cand_doi = cand.doi
            cand_desc = f"Crossref title search: {cand.doi} ({cand.venue}, {cand.year})"
        else:
            hit = s2_lookup(net, e.get("title", ""))
            ext = (hit or {}).get("externalIds") or {}
            if ext.get("DOI"):
                cand_doi = ext["DOI"]
                cand_desc = f"Semantic Scholar: {ext['DOI']}"
        if not e.get("url"):
            fails.append("no DOI and no URL; cannot verify"
                         + (f" (candidate {cand_doi})" if cand_doi else ""))
        else:
            ok, detail = check_url(net, e["url"])
            if ok:
                notes.append(f"no DOI in entry; URL verified ({detail}): {e['url']}")
            else:
                fails.append(f"no DOI and the URL did not resolve ({detail}): {e['url']}")
            if cand_doi:
                notes.append("an archival record with the same title exists -- " + cand_desc)
            else:
                notes.append("no DOI found for this title on Crossref or Semantic Scholar")

    # preprint handling
    if is_preprint_entry(e):
        if not marked_preprint(e):
            fails.append("preprint not marked as a preprint")
        hit = s2_lookup(net, e.get("title", ""))
        pub_doi = ""
        if hit:
            ext = hit.get("externalIds") or {}
            cand_doi = (ext.get("DOI") or "").lower()
            if cand_doi and not cand_doi.startswith(("10.48550", "10.1101", "10.2139")):
                pub_doi = cand_doi
        if not pub_doi:
            cand = crossref_title_search(net, e.get("title", ""), e.get("author", ""))
            if cand and cand.doi and not cand.doi.startswith(("10.48550", "10.1101", "10.2139")):
                pub_doi = cand.doi
        if pub_doi:
            fails.append(f"peer-reviewed version appears to exist: {pub_doi} (cite that instead)")
        else:
            notes.append("no peer-reviewed version found (Crossref + Semantic Scholar)")

    return fails, notes, rec


# ----------------------------------------------------------------------------- report

MARKER = "<!-- AUTOGENERATED SECTION: check_references.py -- edits below the closing marker are kept -->"
END_MARKER = "<!-- END AUTOGENERATED SECTION -->"


def build_report(results, uncited, missing, net, bib_path, tex_path) -> str:
    n = len(results)
    failed = [r for r in results if r["fails"]]
    lines = [MARKER, "",
             "# Reference audit (paper/refs.bib, paper/main.tex)", "",
             f"Generated by `src/check_references.py` on {time.strftime('%Y-%m-%d %H:%M')} local time.",
             f"Sources: `{bib_path}`, `{tex_path}`. APIs: Crossref (polite pool, mailto set), "
             "DataCite, Semantic Scholar.", "",
             "## 1. Metadata verification", "",
             f"- entries in refs.bib: **{n}**",
             f"- entries verified with no mismatch: **{n - len(failed)}**",
             f"- entries with at least one mismatch: **{len(failed)}**",
             f"- uncited entries: **{len(uncited)}**",
             f"- citations with no entry: **{len(missing)}**", ""]
    if net.errors:
        lines += ["API problems during this run:", ""]
        lines += [f"- {e}" for e in sorted(set(net.errors))] + [""]
    lines += ["| key | type | resolved | mismatches |", "|---|---|---|---|"]
    for r in results:
        src = r["record"].source if r["record"] else "-"
        cell = "OK" if not r["fails"] else "<br>".join(x.replace("|", "/") for x in r["fails"])
        lines.append(f"| `{r['key']}` | {r['type']} | {src} | {cell} |")
    lines.append("")
    if uncited:
        lines += ["### Uncited entries", ""] + [f"- `{k}`" for k in uncited] + [""]
    else:
        lines += ["### Uncited entries", "", "None.", ""]
    if missing:
        lines += ["### Citations without an entry", ""] + [f"- `{k}`" for k in missing] + [""]
    else:
        lines += ["### Citations without an entry", "", "None.", ""]
    lines += ["### Notes per entry", ""]
    for r in results:
        if r["notes"]:
            lines.append(f"- `{r['key']}`: " + "; ".join(r["notes"]))
    lines += ["", END_MARKER, ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--bib", default=str(root / "paper" / "refs.bib"))
    ap.add_argument("--tex", default=str(root / "paper" / "main.tex"))
    ap.add_argument("--out", default=str(root / "research" / "REFERENCE_AUDIT.md"))
    ap.add_argument("--cache", default=str(root / "research" / ".refcache.json"))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    bib_path, tex_path, out_path = Path(args.bib), Path(args.tex), Path(args.out)
    entries = parse_bib(read_text(bib_path))
    tex = read_text(tex_path)
    cites = cited_keys(tex)

    keys = [e["_key"] for e in entries]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    uncited = [k for k in keys if k not in cites]
    missing = sorted(k for k in cites if k not in keys)

    net = Net(Path(args.cache), refresh=args.refresh, offline=args.offline)
    results = []
    for i, e in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {e['_key']}", flush=True)
        fails, notes, rec = check_entry(net, e)
        results.append({"key": e["_key"], "type": e["_type"], "fails": fails,
                        "notes": notes, "record": rec})
    net.save()

    report = build_report(results, uncited, missing, net, bib_path.name, tex_path.name)

    # preserve hand-written sections that follow the generated block
    tail = ""
    if out_path.exists():
        old = read_text(out_path)
        if END_MARKER in old:
            tail = old.split(END_MARKER, 1)[1].lstrip("\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(report + ("\n" + tail if tail else ""))

    n_fail = sum(1 for r in results if r["fails"])
    print(f"\nentries {len(entries)}; mismatched {n_fail}; uncited {len(uncited)}; "
          f"missing {len(missing)}; duplicate keys {len(dupes)}; API errors {len(set(net.errors))}")
    print(f"report written to {out_path}")
    bad = n_fail or uncited or missing or dupes or net.errors
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
