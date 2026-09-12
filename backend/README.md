# backend/

FastAPI service and async job workers. Planned layout (see
`TODO/ROADMAP.md` Sprint 13 for the API/job-queue build-out):

```
backend/
  app/           # FastAPI app, routers, pydantic schemas
  workers/       # Celery/RQ job workers (generation, RL, docking)
  registry/      # deploy-side model version selection
```

Not implemented yet — this is a placeholder.
