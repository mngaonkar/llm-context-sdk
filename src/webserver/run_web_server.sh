#!/bin/bash

uvicorn webserver.server:app --host "0.0.0.0" --port 8000 --reload