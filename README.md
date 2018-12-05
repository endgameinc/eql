# Event Query Language
See https://eql.readthedocs.io for documentation

![](docs/_static/eql-whoami.jpg "What is EQL")
Browse a [library of EQL analytics](https://eqllib.readthedocs.io)

# Getting Started

The EQL module current supports Python 2.7 and 3.5+. Assuming a supported Python version is installed, run the command:

```console
$ pip install eql
```

If Python is configured and already in the PATH, then ``eql`` will be readily available, and can be checked by running the command:

```console
$ eql --version
eql 0.6.1
```

From there, try a [sample json file](docs/_static/example.json) and test it with EQL.

```console
$ eql query -f example.json "process where process_name == 'explorer.exe'"
{"command_line": "C:\\Windows\\Explorer.EXE", "event_subtype_full": "already_running", "event_type_full": "process_event", "md5": "ac4c51eb24aa95b77f705ab159189e24", "opcode": 3, "pid": 2460, "ppid": 3052, "process_name": "explorer.exe", "process_path": "C:\\Windows\\explorer.exe", "serial_event_id": 34, "timestamp": 131485997150000000, "unique_pid": 34, "unique_ppid": 0, "user_domain": "research", "user_name": "researcher"}
```

# Next Steps
- Browse a [library of EQL analytics](https://eqllib.readthedocs.io)
- Check out the [query guide](https://eql.readthedocs.io/en/latest/query-guide/index.html) for a crash course on writing EQL queries
- View usage for the [CLI](https://eql.readthedocs.io/en/latest/cli.html)
- Explore the [API](https://eql.readthedocs.io/en/latest/api/index.html) for advanced usage or incorporating EQL into other projects
