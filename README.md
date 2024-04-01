# Shoulder2Shoulder

A web application to foster community engagement and fight the loneliness pandemic.

## Virtual Environments

Both front and back ends use conda virtual environments.

<pre>
```
cd frontend
conda create --file=environment.yml
conda activate frontend
```
</pre>

<pre>
```
cd backend
conda create --file=environment.yml
conda activate backend
```
</pre>

## Pre-Commit Checklist

- backend: run flake8 to check for errors

<pre>
```
cd backend
flake8
```
</pre>

- frontend: format and lint

<pre>
```
cd frontend
npm run lint
npm run format
```
</pre>
