.PHONY: up down build logs test test-unit test-eval sample-data eval eval-baseline shell

up:
	docker compose up --build

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f backend

# --- Local (non-docker) dev commands: run from repo root ---

sample-data:
	python3 scripts/generate_sample_data.py --out backend/eval/data --count 30

test:
	cd backend && python3 -m pytest -v

test-unit:
	cd backend && python3 -m pytest -v -k "not eval"

test-eval:
	cd backend && python3 -m pytest -v tests/test_eval.py

eval:
	cd backend && python3 -m eval.run_eval

eval-baseline:
	cd backend && python3 -m eval.run_eval --save-baseline

shell:
	docker compose exec backend /bin/bash
