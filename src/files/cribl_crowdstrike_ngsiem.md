# CrowdStrike Falcon Next-Gen SIEM Pack
----

## About this Pack

This pack is built as a complete SOURCE + DESTINATION solution (identified by the IO suffix). Data collection and delivery happen entirely within the pack's context, eliminating the need to connect it to globally defined Sources and Destinations. 

This Cribl Pack is designed to streamline the integration of common Cribl data sources with the CrowdStrike Falcon Next-Gen SIEM Platform. It provides pre-configured sources, destinations, pipelines, and routes to format and enrich data from various sources, ensuring compatibility with Next-Gen SIEM ingestion requirements/parsers. The pack simplifies the process of collecting, normalizing, and forwarding security events, enabling efficient analysis and threat detection within the CrowdStrike Next-Gen SIEM platform.


Key features include:
- Support for common Cribl sources including Palo Alto Networks Firewall, Cisco ASA, Fortinet Fortigate, and Windows Security Event logs.
- Data filtering and reduction that is compatible with default Next-Gen SIEM parsers.
- Easy-to-customize configurations that can adapt to specific use cases or environments.

This pack is ideal for security teams looking to leverage Cribl's data processing capabilities to enhance their Crowdstrike Next-Gen SIEM workflows.

This pack is meant to include the full workflow (source->pipeline->destination) so additional sources will be available with updates. 

## Deployment

This pack is configured by default to use the NGSIEM Destinations included in the Pack. To use a different Destination, you must either:
* Add the additional Destination(s) into the Pack
* Update the pack's routes to specify your desired Destination.

All sources are *disabled* by default to prevent unnecessary port conflicts. Variables referenced in this Pack can be updated under Knowledge > Variables.

### Configure the Next-Gen SIEM Destinations

Before configuring the pre-built Next-Gen SIEM Destinations, you must create them in the Next-Gen SIEM platform:
* For each Pack-supported source, create a HEC/HTTP Event Data Connector in the Next-Gen SIEM Console.
* Use the following Parsers for each data source (these Parsers have been tested as being compatible with the Pack):
  * Palo Alto Firewall: *paloalto-ngfw*
  * Cisco ASA Firewall: *cisco-asa*
  * Fortinet Fortigate Firewall: *fortinet-fortigate*
  * Cisco Identity Services Engine: *cisco-ise*
  * Citrix Netscaler: *citrix-netscaler-adc*
  * Windows Security logs: *microsoft-windows*
* Generate an API Key for each Data Connector
* Update the corresponding Pack Next-Gen SIEM Destination with the correct *API URL* and *authentication token*

[Cribl CrowdStrike Falcon Next-Gen SIEM documentation](https://docs.cribl.io/stream/destinations-crowdstrike-next-gen-siem/#configuring-a-crowdstrike-falcon-next-gen-siem-destination)


### Configure Sources

#### Palo Alto Networks Firewall


**Source: Syslog**

Variables:
- `crowdstrike_ngsiem_pan_syslog_udp_port`
- `crowdstrike_ngsiem_pan_syslog_tcp_port`

Lookup:
- `cribl_crowdstrike_ngsiem_pan_device_info.csv`: Use if your PAN device(s) use a non-GMT TZ. Enable the Lookup function in the `cribl_crowdstrike_ngsiem_pan_traffic` pipeline as well. 

Configure the syslog sender on your Palo Alto Networks appliance to forward to the port/protocol specified. The default port is set to `20000` which should be open by default on any Cribl-managed cloud worker.

#### Cisco ASA

**Source: Syslog**

Variables:
- `crowdstrike_ngsiem_cisco_asa_udp_syslog_port`
- `crowdstrike_ngsiem_cisco_asa_tcp_syslog_port`
- `cribl_crowdstrike_ngsiem_asa_drop_events`: Set to `true` to drop events listed in the lookup below (defaults to `false`)

Lookup:
- `cribl_crowdstrike_ngsiem_asa_drops.csv`: Contains a list of ASA codes to drop from the event stream.

Configure the syslog sender on your Cisco ASA appliance to forward to the port/protocol specified. The default port is set to `20001` which should be open by default on any Cribl-managed cloud worker.

#### Fortigate Fortinet Firewall

**Source: Syslog**

Variables:
- `crowdstrike_ngsiem_fortigate_fortinet_udp_syslog_port`
- `crowdstrike_ngsiem_fortigate_fortinet_udp_syslog_port`

Configure the syslog sender on your Fortigate Fortinet appliance to forward to the port/protocol specified. The default port is set to `20002` which should be open by default on any Cribl-managed cloud worker.

#### Citrix Netscaler

**Source: Syslog**

Variables:
- `crowdstrike_ngsiem_citrix_netscaler_udp_syslog_port`
- `crowdstrike_ngsiem_citrix_netscaler_tcp_syslog_port`

Configure the syslog sender on your Citrix Netscaler appliance to forward to the port/protocol specified. The default port is set to `20004` which should be open by default on any Cribl-managed cloud worker.

#### Cisco ISE

**Source: Syslog**

Variables:
- `crowdstrike_ngsiem_cisco_ise_udp_syslog_port`
- `crowdstrike_ngsiem_cisco_ise_tcp_syslog_port`

Configure the syslog sender on your Cisco ISE appliance to forward to the port/protocol specified. The default port is set to `20005` which should be open by default on any Cribl-managed cloud worker.


#### Windows Event log

**Source: Cribl HTTP** (events sourced from Cribl Edge)

Variables:
- `cribl_crowdstrike_windows_events_http_port`

Lookup:
- `cribl_crowdstrike_ngsiem_retained_windows_eventIDs.csv`: Contains a list of *allowed* Windows EventID's - all others will be dropped.  

This source is designed to receive Windows Event Logs from Cribl Edge via default port `20003`. CrowdStrike Falcon Next-Gen SIEM *only* accepts Windows Event Logs in XML format. Ensure that your Cribl Edge deployments send Windows Event's in XML format vs the default JSON.

## Upgrades

Upgrading certain Cribl Packs using the same Pack ID can have unintended consequences. See [Upgrading an Existing Pack](https://docs.cribl.io/stream/packs#upgrading) for details.

## Release Notes

### Version 1.0.0
Initial release

## Contributing to the Pack

To contribute to the Pack, please connect with us on [Cribl Community Slack](https://cribl-community.slack.com/). You can suggest new features or offer to collaborate.

## License
This Pack uses the following license: [Apache 2.0](https://github.com/criblio/appscope/blob/master/LICENSE).

