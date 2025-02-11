CREATE SCHEMA public AUTHORIZATION azure_pg_admin;


--Table for cases of patient of tenant/practice
CREATE TABLE public.dev_cases (
	id serial4 NOT NULL, -- unique identifier for this table as a serial number for table
	case_id varchar NULL, -- refers to a patient case unique identifier
	"data" jsonb NOT NULL, --in detail case related information in JSONB individually 
	tenant_id varchar NULL, -- refers to a tenant/practice unique identifier
	CONSTRAINT dev_cases_case_id_key UNIQUE (case_id),
	CONSTRAINT dev_cases_pkey PRIMARY KEY (id)
);



--Table for locations of various tenants/practice
CREATE TABLE public.dev_locations (
	id varchar NOT NULL, --unique identifier for this table as a serial number for table
	"name" varchar NULL, -- location name
	dibbs_name varchar NULL, -- tenant/practice names for dibbs software
	tenant_id varchar NOT NULL,-- refers to a tenant/practice unique identifier
	street varchar NULL, --street of practice location (part of address)
	city varchar NULL, --city of practice location (part of address)
	state varchar NULL, --state of practice location (part of address)
	country varchar NULL, --country of practice location (part of address)
	postal_code varchar NULL, --postal code of practice location (part of address)
	phone varchar NULL, --phone number of tenant/practice
	contact_name varchar NULL, --contact name of tenant/practice
	office_phone_number varchar NULL, --office phone number of the tenant/practice
	email varchar NULL, --email of the tenant/practice
	state_or_region varchar NULL, --need to check the difference
	intra_oral_scanner varchar NULL, --scanner used at tenant/practice location
	office_hours varchar NULL, --office hours of tenant/practice
	rush_rate numeric NULL, --rate for rushed cases in the tenant/practice
	standard_rate numeric NULL, --standard rate for usual cases in the tenant/practice
	deactivated bool DEFAULT false NULL, --practice deactivated? (TRUE or FALSE)
	CONSTRAINT dev_locations_pkey PRIMARY KEY (tenant_id, id)
);



--Table for various orders
CREATE TABLE public.dev_orders (
	order_id varchar NOT NULL, --unique identifier of the order to identify who the order is for
	tenant_id varchar NULL, -- refers to a tenant/practice unique identifier
	placed_on varchar NULL, --date when the order was placed
	placed_by varchar NULL, --name/id of the user who placed the order
	packed_on varchar NULL, --date when the order was packed
	status varchar NULL, --current status of the order 
	tracking varchar NULL, --tracking number of the order
	shipped_on varchar NULL, --date when the order was shipped
	location_id varchar NULL, --unique identifier of location where the order is to be shipped
	expedite_shipping bool DEFAULT false NULL, --is the order expedited? (TRUE or FALSE)
	CONSTRAINT dev_orders_pkey PRIMARY KEY (order_id)
);





--Table for preference of the practice/tenant
CREATE TABLE public.dev_preferences (
	id serial4 NOT NULL, --unique identifier for this table as a serial number for table
	tenant_id varchar NULL, -- refers to a tenant/practice unique identifier
	archwire varchar NULL, -- preference for archwire type
	hooks varchar NULL, -- preference for hooks 
	tray_sectioning varchar NULL, --refers to tray_sectioning 
	printing_location varchar NULL, --location where the manufacturing takes place
	CONSTRAINT dev_preferences_pkey PRIMARY KEY (id),
	CONSTRAINT dev_preferences_tenant_id_key UNIQUE (tenant_id)
);





--Table for various tenants/practice
CREATE TABLE public.dev_tenants (
	id serial4 NOT NULL, --unique identifier for this table as a serial number for table
	tenant_id varchar NULL, -- refers to a tenant/practice unique identifier
	practice_name varchar NULL, -- name of the practice/tenant
	account_id varchar NULL, --CRM Customer ID
	CONSTRAINT dev_tenants_pkey PRIMARY KEY (id),
	CONSTRAINT dev_tenants_tenant_id_key UNIQUE (tenant_id)
);




--Table for users of the system
CREATE TABLE public.dev_users (
	id varchar NOT NULL, --unique identifier for this table as a serial number for table
	tenant_id varchar NULL, -- refers to a tenant/practice unique identifier
	first_name varchar NULL, --first name of the user
	last_name varchar NULL, --last name of the user
	phone varchar NULL, --phone number of the user
	is_admin bool NULL, -- is the user admin? (TRUE or FALSE)
	is_doctor bool NULL, -- is the user doctor or not? (TRUE or FALSE)
	email varchar NULL, --email id of the user
	restricted_location_ids varchar NULL, --what all locations are restricted for the particular user
	azure_b2c_id varchar NULL, --azure id of the user
	active bool NULL, --is the user active? (TRUE or FALSE)
	CONSTRAINT dev_users_pkey PRIMARY KEY (id)
);




--table for workflow and events status
CREATE TABLE public.dev_workflow_events (
	id serial4 NOT NULL, --unique identifier for this table as a serial number for table
	message_id varchar NULL, --id for unique message from one workflow to other 
	case_id varchar NULL, -- refers to a patient case unique identifier
	kind varchar NULL, --Action taken on case (step completed/failed/need for modification)
	"version" varchar NULL, --internal id (NOT NEEDED)
	who varchar NULL, --technician name /technician id who performed workflow event
	occurred_at varchar NULL, --date when workflow event was completed
	recorded_at varchar NULL,  --date when workflow event was recorded for completion (same date as occured_at)
	lab_status varchar NULL, -- case current state
	tenant_id varchar NULL, -- refers to a tenant/practice unique identifier
	CONSTRAINT dev_workflow_events_message_id_key UNIQUE (message_id),
	CONSTRAINT dev_workflow_events_pkey PRIMARY KEY (id)
);

CREATE INDEX dev_dwe_caseid ON public.dev_workflow_events USING btree (case_id);
CREATE INDEX idx_dev_workflow_events_tenant_kind_occurred_at ON public.dev_workflow_events USING btree (tenant_id, kind, occurred_at);



--used to see difference of dates in approval from sent_for_approval to approved_on
CREATE TABLE public.difference_of_approval_dates (
	case_id int4 NULL,
	sent_for_approval_on timestamp NULL,
	approved_on timestamp NULL,
	date_difference int4 NULL
);


CREATE TABLE public.holidays (
	calendardate date NOT NULL,
	description varchar(30) NULL,
	CONSTRAINT pk_holidays PRIMARY KEY (calendardate)
);


--Table for descirption of ordered items
CREATE TABLE public.dev_orders_items (
	order_id varchar NOT NULL, --unique identifier for each order
	sku varchar NOT NULL, --stock keeping unit of a order
	quantity int4 NULL, --total order quantity (unit : in packs of5/10 ....) 
	price numeric NULL, --price of the order
	CONSTRAINT dev_orders_items_pkey PRIMARY KEY (order_id, sku),
	CONSTRAINT dev_orders_items_order_id_fkey FOREIGN KEY (order_id) REFERENCES public.dev_orders(order_id)
);


--Table for preferences of users of the system
CREATE TABLE public.dev_users_preferences (
	id varchar NULL,--unique identifier for this table as a serial number for table
	preferences jsonb NULL, --preference of a user in JSONB format
	CONSTRAINT dev_users_preferences_id_key UNIQUE (id),
	CONSTRAINT dev_users_preferences_id_fkey FOREIGN KEY (id) REFERENCES public.dev_users(id) ON DELETE CASCADE
);

