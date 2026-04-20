# Challenge Day Checklist (60 Seconds)

Use this immediately before any right-column evaluation submission.

---

## A. Dataset and output sanity

- [ ] Output generated from `Submission_Levels` (not `Public_Levels`).
- [ ] Correct level selected (1/2/3 matches output file).
- [ ] Output file is plain text and non-empty.
- [ ] Each line contains exactly one valid ID.

---

## B. Session ID integrity

- [ ] Session ID is from the exact run that generated this output file.
- [ ] Session ID format is `{TEAM_NAME}-{ULID}`.

---

## C. Source zip integrity

- [ ] Zip includes required `.py` files.
- [ ] Zip includes `requirements.txt`.
- [ ] Zip includes `.env.example` (no real `.env`).
- [ ] Zip includes `README` with run instructions.
- [ ] No secrets, `.venv`, caches, or irrelevant large folders.

---

## D. One-shot confirmation

- [ ] This is the intended final one-shot evaluation submission for this level.
- [ ] Team agrees to submit this level now.

---

## E. Quick commands (optional)

```bash
file -bi <output_file>
wc -l <output_file>
cat <output_file>
unzip -l <source_zip>
```

If any check fails: stop upload, regenerate, and revalidate.
