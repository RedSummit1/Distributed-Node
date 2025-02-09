from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import logging
import uuid


class RegisterObject(BaseModel):
    data: Dict[str, Any]


logger = logging.getLogger(f"{__name__}.server")
server = None


def create_server(node=None, host="0.0.0.0", admin_rest_port=8080):
    global server
    app = FastAPI()
    logger.info("Instanciate FastAPI application")
    _set_up_routes(node=node, app=app)

    # Log routes
    for route in app.routes:
        logger.info(f"Route: {route.path} [{route.name}]")

    try:
        config = uvicorn.Config(
            app,
            host=host,
            port=admin_rest_port,
            log_level="error",
        )

        server = uvicorn.Server(config)
        logger.info("Instanciate server application")

        if not server:
            raise ValueError("server returned None type")

    except ValueError as e:
        logger.error("Falied to instanciate server application")

    return server


def _set_up_routes(node=None, app=None):
    logger.info("Setting up http routes")
    if not app:
        raise ValueError("Fast application returned None type")
    if not node:
        raise ValueError("Node object returned None type")

    @app.get("/ping")
    async def ping():
        """
        Pings a node

        Returns:
            A json of the status
        """
        return {"status": "ok"}

    @app.post("/submitJob")
    async def submit_job(auth_code):
        """
        Submit a job to the providing node from Admin Portal

        Args:
            param1: Authentication Code

        Returns:
            Returns a job id as UUID to Admin Portal
        """
        node.check_auth(auth_code)
        job_id = str(uuid.uuid4())
        node.job_schedule.update(
            {
                job_id: {
                    "Authorization Code": auth_code,
                    "STATUS": "NOT STARTED",
                },
            }
        )

        return {"Job ID": job_id}

    @app.post("/registerJob")
    async def register_job(request: RegisterObject):
        """
        Submit a job to the providing node from Admin Portal

        Args:
            param1: Authentication Code
            param2: Job ID

        Returns:
            Returns a job id as UUID to Admin Portal
        """
        node.job_schedule.update(
            {
                request.data["job_id"]: {
                    "Authorization Code": request.data["auth_code"],
                    "STATUS": "NOT STARTED",
                }
            }
        )
        return request.data

    @app.get("/getJobSchedule")
    async def get_jobSchedule():
        """
        Get the job schedule

         Returns:
             Returns the complete jobSchedule
        """

        return {"jobSchedule": "<JobSchedule>"}

    @app.get("/getJob/{job_id}")
    async def get_job():
        """
        Get an individual job from the JobSchedule

         Args:
             param1: Job ID

         Returns:
            An individual job

        """
        return {"Job ID": "<Job>"}

    @app.get("/info")
    async def info():
        """
        Get an individual job from the JobSchedule

        Returns:
             Information about the node
        """
        return {"<keys>": "<values>"}
